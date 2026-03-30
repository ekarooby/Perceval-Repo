import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components as comp
from pyqsp import angle_sequence
from pyqsp.poly import PolyTaylorSeries

# ============================================================
# PART 1: Reproduce pyqsp code exactly (your working code)
# ============================================================

N_approx = 100
polydeg = 15

def step_surrogate(u):
    return (2.0 / np.pi) * np.arctan(N_approx * np.pi * u)

def step_theta(theta):
    return np.where(theta < 0.0, -1.0, 1.0)

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
print(f"Polynomial degree : {polydeg}")
print(f"Parity            : {parity}")
print(f"L                 : {L}")
print(f"QSP phases        : {phiset}")

# ============================================================
# PART 2: Ground truth = poly(u) exactly as in your code
#
# theta ∈ [-pi, pi]  is the plot domain (paper x-axis)
# u = theta / pi ∈ [-1, 1]  is the polynomial domain
# Classical output = poly(u) = poly(theta/pi)
# ============================================================

theta_grid = np.linspace(-np.pi, np.pi, 300)
u_grid     = theta_grid / np.pi

# Ground truth: exactly as your working pyqsp code
target_vals    = step_theta(theta_grid)
surrogate_vals = step_surrogate(u_grid)
approx_vals    = poly(u_grid)   # this is the green curve

# ============================================================
# PART 3: PIC circuit
#
# Input to PIC: u ∈ [-1, 1]  (same as poly domain)
# Signal unitary uses u directly: Rx(arccos(u))
# We test all output formulas to find which matches poly(u)
# ============================================================

def PS_mat(phi):
    return np.array([
        [np.exp(1j * phi), 0],
        [0,                1]
    ], dtype=complex)

def Rx_mat(u):
    sx = np.sqrt(max(1 - u**2, 0))
    return np.array([
        [u,      1j * sx],
        [1j * sx, u     ]
    ], dtype=complex)

def build_qsp_pic(phiset, u_val, L):
    sx = np.sqrt(max(1 - float(u_val)**2, 0))
    Rx = np.array([
        [float(u_val), 1j * sx     ],
        [1j * sx,      float(u_val)]
    ], dtype=complex)
    circuit = pcvl.Circuit(2, name=f"QSP_L{L}")
    circuit.add(0, comp.PS(float(phiset[0])))
    for j in range(1, L + 1):
        circuit.add((0, 1), pcvl.Unitary(pcvl.Matrix(Rx)))
        circuit.add(0, comp.PS(float(phiset[j])))
    return circuit

def get_all_pic_outputs(phiset, u_val, L):
    circuit = build_qsp_pic(phiset, u_val, L)
    U = circuit.compute_unitary()
    psi = U @ np.array([1.0, 0.0])
    return {
        "2Im" : 2.0 * np.imag(psi[0]),
        "2Re" : 2.0 * np.real(psi[0]),
        "Z"   : abs(psi[0])**2 - abs(psi[1])**2,
        "Im"  : np.imag(psi[0]),
        "Re"  : np.real(psi[0]),
    }

# ============================================================
# PART 4: Sanity check at u=0.5
# poly(0.5) is ground truth — find which PIC formula matches
# ============================================================

u_test   = 0.5
poly_ref = float(poly(np.array([u_test]))[0])
outs     = get_all_pic_outputs(phiset, u_test, L)

print("\n========== Sanity Check at u=0.5 ==========")
print(f"  poly(0.5) ground truth : {poly_ref:.6f}")
print(f"  PIC 2*Im(psi[0])       : {outs['2Im']:.6f}")
print(f"  PIC 2*Re(psi[0])       : {outs['2Re']:.6f}")
print(f"  PIC Z = p0-p1          : {outs['Z']:.6f}")
print(f"  PIC Im(psi[0])         : {outs['Im']:.6f}")
print(f"  PIC Re(psi[0])         : {outs['Re']:.6f}")
print("  Match with poly(0.5)?")
print("============================================")

# ============================================================
# PART 5: Full sweep — compute all PIC formulas
# ============================================================

n = len(theta_grid)
f_2im = np.zeros(n)
f_2re = np.zeros(n)
f_z   = np.zeros(n)
f_im  = np.zeros(n)
f_re  = np.zeros(n)

print(f"\nSweeping {n} points...")
for i, u in enumerate(u_grid):
    outs_i   = get_all_pic_outputs(phiset, u, L)
    f_2im[i] = outs_i["2Im"]
    f_2re[i] = outs_i["2Re"]
    f_z[i]   = outs_i["Z"]
    f_im[i]  = outs_i["Im"]
    f_re[i]  = outs_i["Re"]
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{n} done")

print("Sweep complete.")

# ============================================================
# PART 6: MSE of each formula vs poly(u)
# ============================================================

mse_2im = np.mean((f_2im - approx_vals) ** 2)
mse_2re = np.mean((f_2re - approx_vals) ** 2)
mse_z   = np.mean((f_z   - approx_vals) ** 2)
mse_im  = np.mean((f_im  - approx_vals) ** 2)
mse_re  = np.mean((f_re  - approx_vals) ** 2)

print("\n========== MSE vs poly(u) ground truth ==========")
print(f"  2*Im : {mse_2im:.2e}")
print(f"  2*Re : {mse_2re:.2e}")
print(f"  Z    : {mse_z:.2e}")
print(f"  Im   : {mse_im:.2e}")
print(f"  Re   : {mse_re:.2e}")
print("  Smallest = correct PIC output formula")
print("=================================================")

# ============================================================
# PART 7: Plot — exactly like your pyqsp code + PIC overlay
# The green curve (approx_vals = poly(u)) is ground truth
# PIC output should sit on top of the green curve
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Photonic QSP PIC  STEP function  L={L}", fontsize=13)

# Left: reproduce your pyqsp plot + best PIC formula
ax = axes[0]
ax.plot(theta_grid, target_vals,    label="Target STEP",           linewidth=2, color='blue')
ax.plot(theta_grid, surrogate_vals, ":", label="arctan surrogate", linewidth=1.5, color='orange')
ax.plot(theta_grid, approx_vals,    "--", label="poly(u) classical", linewidth=2, color='green')
ax.plot(theta_grid, f_2im,          ".", ms=2, label=f"PIC 2Im MSE={mse_2im:.2e}", color='red')
ax.plot(theta_grid, f_im,           ".", ms=2, label=f"PIC Im  MSE={mse_im:.2e}",  color='purple')
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.3, 1.3])
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"], fontsize=12)
ax.set_xlabel(r"$\theta$", fontsize=12)
ax.set_ylabel("Function value", fontsize=12)
ax.set_title(f"STEP L={L}  green=poly(u)  PIC must match green", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Right: residuals vs poly(u)
ax2 = axes[1]
ax2.plot(theta_grid, f_2im - approx_vals, 'r-',      lw=1.5, label=f"2Im MSE={mse_2im:.2e}")
ax2.plot(theta_grid, f_2re - approx_vals, 'b-',      lw=1.5, label=f"2Re MSE={mse_2re:.2e}")
ax2.plot(theta_grid, f_z   - approx_vals, 'm-',      lw=1.5, label=f"Z   MSE={mse_z:.2e}")
ax2.plot(theta_grid, f_im  - approx_vals, 'g-',      lw=1.5, label=f"Im  MSE={mse_im:.2e}")
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks([-np.pi, 0, np.pi])
ax2.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"], fontsize=12)
ax2.set_xlabel(r"$\theta$", fontsize=12)
ax2.set_ylabel("Residual vs poly(u)", fontsize=12)
ax2.set_title("Residuals vs poly(u)  smallest = correct formula", fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pic_qsp_step_L15_formula_final.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: pic_qsp_step_L15_formula_final.png")