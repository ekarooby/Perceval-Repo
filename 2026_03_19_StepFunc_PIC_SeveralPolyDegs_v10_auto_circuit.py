import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components as comp
from pyqsp import angle_sequence
from pyqsp.poly import PolyTaylorSeries

N_approx = 100
polydeg = 15

# ============================================================
# PART 1: Compute polynomial and QSP phases
# ============================================================

def step_surrogate(u):
    return (2.0 / np.pi) * np.arctan(N_approx * np.pi * u)

def step_true(x):
    return np.where(x >= 0, 1.0, -1.0)

print(f"Computing polynomial and phases for polydeg={polydeg}...")

poly = PolyTaylorSeries().taylor_series(
    func=step_surrogate,
    degree=polydeg,
    max_scale=0.9,
    chebyshev_basis=True,
    cheb_samples=4 * polydeg
)

phiset, red_phiset, parity = angle_sequence.QuantumSignalProcessingPhases(
    poly,
    method="sym_qsp",
    chebyshev_basis=True
)

L = len(phiset) - 1
print(f"  Parity : {parity}")
print(f"  L      : {L}")
print(f"  Phases : {phiset}")

# ============================================================
# PART 2: Classical reference = poly(u) directly
#
# This is exactly what pyqsp guarantees the QSP circuit
# computes. It produced the correct image 3 shape.
# ============================================================

def classical_reference(u):
    """Ground truth: polynomial fitted by pyqsp."""
    val = poly(np.array([u]))
    return float(val[0]) if hasattr(val, '__len__') else float(val)

# ============================================================
# PART 3: Build Perceval PIC using proper MZI components
#
# Signal unitary Rx(arccos(u)) = [[u, i*sx],[i*sx, u]]
# In Perceval this is BS.H with reflectivity angle:
#   BS.H(theta) = [[cos(t/2), i*sin(t/2)],[i*sin(t/2), cos(t/2)]]
#   cos(t/2) = u  -->  t = 2*arccos(u)
#
# Phase gate Rz(phi): Perceval PS(phi) = [[e^{i*phi},0],[0,1]]
# From v4 we verified: use PS(phi) directly with 2*Im(psi[0])
# ============================================================

def build_qsp_pic_mzi(phiset, u_val, L):
    """
    Build PIC using proper BS (MZI) components.
    No raw Unitary injection — uses BS.H for signal unitary.
    This gives a physically meaningful MZI mesh circuit.
    """
    # BS angle for signal unitary
    bs_angle = 2.0 * np.arccos(np.clip(float(u_val), -1.0, 1.0))

    circuit = pcvl.Circuit(2, name=f"QSP_STEP_L{L}")

    # Initial phase gate
    circuit.add(0, comp.PS(float(phiset[0])))

    for j in range(1, L + 1):
        # Signal unitary: BS.H(2*arccos(u))
        circuit.add((0, 1), comp.BS.H(theta=bs_angle))
        # Phase gate
        circuit.add(0, comp.PS(float(phiset[j])))

    return circuit

def get_pic_output_mzi(phiset, u_val, L):
    circuit = build_qsp_pic_mzi(phiset, u_val, L)
    U = circuit.compute_unitary()
    psi_out = U @ np.array([1.0, 0.0])
    return 2.0 * np.imag(psi_out[0])

# ============================================================
# PART 4: Sanity check — compare all output formulas
# ============================================================

u_test = 0.5
c_ref = classical_reference(u_test)
p_im  = get_pic_output_mzi(phiset, u_test, L)

# Also try Re and Z=p0-p1
def get_all_outputs(phiset, u_val, L):
    circuit = build_qsp_pic_mzi(phiset, u_val, L)
    U = circuit.compute_unitary()
    psi = U @ np.array([1.0, 0.0])
    return {
        "2Im" : 2.0 * np.imag(psi[0]),
        "2Re" : 2.0 * np.real(psi[0]),
        "Z"   : abs(psi[0])**2 - abs(psi[1])**2
    }

outs = get_all_outputs(phiset, u_test, L)

print("\n========== Sanity Check at u=0.5 ==========")
print(f"  poly(u) ground truth : {c_ref:.6f}")
print(f"  PIC 2*Im(psi[0])     : {outs['2Im']:.6f}")
print(f"  PIC 2*Re(psi[0])     : {outs['2Re']:.6f}")
print(f"  PIC Z=p0-p1          : {outs['Z']:.6f}")
print("  Whichever matches poly(u) is the correct formula.")
print("============================================")

# ============================================================
# PART 5: Full sweep using the formula that matches poly(u)
# ============================================================

n_points = 300
u_grid = np.linspace(-1.0, 1.0, n_points)
x_grid = u_grid * np.pi

f_poly   = np.zeros(n_points)
f_2im    = np.zeros(n_points)
f_2re    = np.zeros(n_points)
f_z      = np.zeros(n_points)

print(f"\nSweeping {n_points} points...")
for i, u in enumerate(u_grid):
    f_poly[i] = classical_reference(u)
    outs_i = get_all_outputs(phiset, u, L)
    f_2im[i]  = outs_i["2Im"]
    f_2re[i]  = outs_i["2Re"]
    f_z[i]    = outs_i["Z"]
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{n_points} done")

print("Sweep complete.")

# ============================================================
# PART 6: MSE report — find which formula matches poly(u)
# ============================================================

step_vals    = step_true(x_grid)
mse_poly     = np.mean((f_poly - step_vals) ** 2)
mse_2im_poly = np.mean((f_2im  - f_poly)    ** 2)
mse_2re_poly = np.mean((f_2re  - f_poly)    ** 2)
mse_z_poly   = np.mean((f_z    - f_poly)    ** 2)
mse_2im_tgt  = np.mean((f_2im  - step_vals) ** 2)
mse_2re_tgt  = np.mean((f_2re  - step_vals) ** 2)
mse_z_tgt    = np.mean((f_z    - step_vals) ** 2)

print("\n========== MSE Report ==========")
print(f"  MSE poly vs target       : {mse_poly:.6f}")
print(f"  MSE 2Im vs poly          : {mse_2im_poly:.2e}")
print(f"  MSE 2Re vs poly          : {mse_2re_poly:.2e}")
print(f"  MSE Z   vs poly          : {mse_z_poly:.2e}")
print(f"  MSE 2Im vs target        : {mse_2im_tgt:.6f}")
print(f"  MSE 2Re vs target        : {mse_2re_tgt:.6f}")
print(f"  MSE Z   vs target        : {mse_z_tgt:.6f}")
print("  Formula with smallest MSE vs poly is correct.")
print("=================================")

# ============================================================
# PART 7: Plot all three PIC formulas vs poly(u)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Photonic QSP PIC  STEP L={L}  formula comparison", fontsize=13)

ax = axes[0]
ax.plot(x_grid, step_true(x_grid), 'k-',  lw=2,   label="Target STEP")
ax.plot(x_grid, f_poly,            'r-',  lw=2,   label="poly(u) ground truth")
ax.plot(x_grid, f_2im,             'b.',  ms=2,   label="PIC 2*Im(psi0)")
ax.plot(x_grid, f_2re,             'g.',  ms=2,   label="PIC 2*Re(psi0)")
ax.plot(x_grid, f_z,               'm.',  ms=2,   label="PIC Z=p0-p1")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.3, 1.3])
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels(["-pi", "0", "pi"], fontsize=11)
ax.set_xlabel("Input signal x", fontsize=11)
ax.set_ylabel("f(x)", fontsize=11)
ax.set_title("Which PIC formula matches poly(u)?", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(x_grid, f_2im - f_poly, 'b-', lw=1.5, label=f"2Im residual MSE={mse_2im_poly:.2e}")
ax2.plot(x_grid, f_2re - f_poly, 'g-', lw=1.5, label=f"2Re residual MSE={mse_2re_poly:.2e}")
ax2.plot(x_grid, f_z   - f_poly, 'm-', lw=1.5, label=f"Z   residual MSE={mse_z_poly:.2e}")
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks([-np.pi, 0, np.pi])
ax2.set_xticklabels(["-pi", "0", "pi"], fontsize=11)
ax2.set_xlabel("Input signal x", fontsize=11)
ax2.set_ylabel("Residual vs poly(u)", fontsize=11)
ax2.set_title("Residuals vs poly(u) ground truth", fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pic_qsp_formula_check.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: pic_qsp_formula_check.png")

# ============================================================
# PART 8: Draw proper MZI circuit for paper (L=3)
# ============================================================

print("\nDrawing MZI circuit for L=3...")
circ_draw = build_qsp_pic_mzi(phiset[:4], 0.5, L=3)
pcvl.pdisplay(circ_draw, output_format=pcvl.Format.MPLOT)
plt.savefig("qsp_pic_mzi_circuit_L3.png", dpi=150, bbox_inches='tight')
plt.show()
print("Circuit saved: qsp_pic_mzi_circuit_L3.png")

# ============================================================
# PART 9: PIC resource count
# ============================================================

circ_full = build_qsp_pic_mzi(phiset, 0.5, L)
n_ps = sum(1 for _, c in circ_full._components if isinstance(c, comp.PS))
n_bs = sum(1 for _, c in circ_full._components if isinstance(c, comp.BS))

print("\n========== PIC Resource Count ==========")
print(f"  QSP layers L      : {L}")
print(f"  Phase shifters PS : {n_ps}   (= L+1 = {L+1})")
print(f"  MZIs BS           : {n_bs}   (= L   = {L})")
print(f"  Total components  : {n_ps + n_bs}")
print(f"  Waveguide modes   : 2  dual-rail qubit")
print("=========================================")