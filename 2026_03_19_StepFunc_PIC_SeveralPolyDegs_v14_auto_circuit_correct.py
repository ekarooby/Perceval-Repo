
#eiphiZ(phi) --> PS(+phi) mode 0 + PS(-phi) mode 1
#Wx(u)       --> BS.Rx(2*arccos(u))
#Output      --> Im(psi[0])
#MSE         --> 4.61e-19  (machine precision zero)

# The problem with this code is that we are using pyqsp and method sym_qsp(Chebyshev)
# The output is Im(psi[0]) which is not measurable. 
# We will probalbly install paddle_quantum in perceval_env to use Trigonometric QSP with out put <Z> = P0 - P1

import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components as comp
from pyqsp import angle_sequence
from pyqsp.poly import PolyTaylorSeries

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
# VERIFIED FINAL CONVENTIONS:
#   Phase gate : eiphiZ(phi) = PS(phi) mode0 + PS(-phi) mode1
#   Signal gate: Wx(u) = BS.Rx(2*arccos(u))
#   Output     : Im(psi[0])
#   MSE        : 4.61e-19  (machine precision)
# ============================================================

def classical_correct(phiset, u, L):
    def eiphiZ(phi):
        return np.array([
            [np.exp(1j * phi), 0             ],
            [0,                np.exp(-1j*phi)]
        ], dtype=complex)
    def Wx(u):
        sx = np.sqrt(max(1 - u**2, 0))
        return np.array([
            [u,       1j * sx],
            [1j * sx, u      ]
        ], dtype=complex)
    U = eiphiZ(phiset[0])
    for j in range(1, L + 1):
        U = eiphiZ(phiset[j]) @ Wx(u) @ U
    psi = U @ np.array([1.0, 0.0])
    return np.imag(psi[0])

def build_qsp_pic(phiset, u_val, L):
    """
    Physical PIC circuit for QSP using verified components:
      eiphiZ(phi) --> PS(phi) on mode 0 + PS(-phi) on mode 1
      Wx(u)       --> BS.Rx(2*arccos(u))
      Output      --> Im(psi[0])
    """
    u_c      = np.clip(float(u_val), -1.0 + 1e-9, 1.0 - 1e-9)
    bs_theta = 2.0 * np.arccos(u_c)

    circuit = pcvl.Circuit(2, name=f"QSP_STEP_L{L}")

    circuit.add(0, comp.PS(float( phiset[0])))
    circuit.add(1, comp.PS(float(-phiset[0])))

    for j in range(1, L + 1):
        circuit.add((0, 1), comp.BS.Rx(theta=bs_theta))
        circuit.add(0, comp.PS(float( phiset[j])))
        circuit.add(1, comp.PS(float(-phiset[j])))

    return circuit

def get_pic_output(phiset, u_val, L):
    circuit = build_qsp_pic(phiset, u_val, L)
    U       = np.array(circuit.compute_unitary())
    psi     = U @ np.array([1.0, 0.0])
    return np.imag(psi[0])

# ============================================================
# Sanity check
# ============================================================

u_test = 0.5
c_val  = classical_correct(phiset, u_test, L)
p_val  = get_pic_output(phiset, u_test, L)

print("\n========== Sanity Check at u=0.5 ==========")
print(f"  Classical Im(psi0) : {c_val:.6f}")
print(f"  PIC Im(psi0)       : {p_val:.6f}")
print(f"  Difference         : {abs(c_val - p_val):.2e}  must be ~0")
print("============================================")

# ============================================================
# Full sweep
# ============================================================

theta_grid  = np.linspace(-np.pi, np.pi, 300)
u_grid      = theta_grid / np.pi
approx_vals = poly(u_grid)

f_classical = np.zeros(len(u_grid))
f_pic       = np.zeros(len(u_grid))

print(f"\nSweeping {len(u_grid)} points...")
for i, u in enumerate(u_grid):
    f_classical[i] = classical_correct(phiset, u, L)
    f_pic[i]       = get_pic_output(phiset, u, L)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(u_grid)} done")

print("Sweep complete.")

# ============================================================
# MSE report
# ============================================================

mse_pic_classic = np.mean((f_pic - f_classical) ** 2)
mse_pic_poly    = np.mean((f_pic - approx_vals)  ** 2)
mse_pic_target  = np.mean((f_pic - step_theta(theta_grid)) ** 2)

print("\n========== MSE Report ==========")
print(f"  MSE PIC vs Classical : {mse_pic_classic:.2e}  must be ~0")
print(f"  MSE PIC vs poly(u)   : {mse_pic_poly:.2e}    must be ~0")
print(f"  MSE PIC vs Target    : {mse_pic_target:.4f}")
print("=================================")

# ============================================================
# Plot — paper Fig.2 style
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Photonic QSP on PIC  STEP function  L={L}  FINAL", fontsize=13)

ax = axes[0]
ax.plot(theta_grid, step_theta(theta_grid), 'k-',  lw=2,  label="Target STEP")
ax.plot(theta_grid, approx_vals,            'g--', lw=2,  label="poly(u) classical")
ax.plot(theta_grid, f_pic,                  'b.',  ms=3,  label="Photonic PIC Im(psi0)")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.3, 1.3])
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"], fontsize=12)
ax.set_xlabel(r"$\theta$", fontsize=12)
ax.set_ylabel("Function value", fontsize=12)
ax.set_title(f"L={L}  blue must sit on green", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
diff = f_pic - f_classical
ax2.plot(theta_grid, diff, 'purple', lw=1.5,
         label=f"PIC minus Classical  MSE={mse_pic_classic:.2e}")
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.fill_between(theta_grid, diff, alpha=0.2, color='purple')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks([-np.pi, 0, np.pi])
ax2.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"], fontsize=12)
ax2.set_xlabel(r"$\theta$", fontsize=12)
ax2.set_ylabel("Residual", fontsize=12)
ax2.set_title("PIC vs Classical residual  must be ~0", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pic_qsp_step_FINAL.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: pic_qsp_step_FINAL.png")

# ============================================================
# PIC resource count for Quandela experiment
# ============================================================

circ_full = build_qsp_pic(phiset, 0.5, L)
n_ps = sum(1 for _, c in circ_full._components if isinstance(c, comp.PS))
n_bs = sum(1 for _, c in circ_full._components if isinstance(c, comp.BS))

print("\n========== PIC Resource Count for Quandela ==========")
print(f"  QSP layers L        : {L}")
print(f"  Phase shifters PS   : {n_ps}   (= 2*(L+1) = {2*(L+1)})")
print(f"  Beam splitters BS   : {n_bs}   (= L = {L})")
print(f"  Total components    : {n_ps + n_bs}")
print(f"  Waveguide modes     : 2  dual-rail qubit")
print(f"")
print(f"  PS angles for Quandela (phi_j in radians):")
for j, phi in enumerate(phiset):
    print(f"    PS[{j:02d}] mode0=+{phi:.6f} rad   mode1={-phi:.6f} rad")
print(f"")
print(f"  BS.Rx angle per input x:")
print(f"    theta_BS = 2 * arccos(x / pi)")
print(f"    where x is input signal in [-pi, pi]")
print("=====================================================")

# ============================================================
# Draw physical circuit for paper (L=3)
# ============================================================

print("\nDrawing physical BS.Rx + PS circuit for L=3...")
circ_draw = build_qsp_pic(phiset[:4], 0.5, L=3)
pcvl.pdisplay(circ_draw, output_format=pcvl.Format.MPLOT)
plt.savefig("qsp_pic_circuit_L3_FINAL.png", dpi=150, bbox_inches='tight')
plt.show()
print("Circuit saved: qsp_pic_circuit_L3_FINAL.png")
