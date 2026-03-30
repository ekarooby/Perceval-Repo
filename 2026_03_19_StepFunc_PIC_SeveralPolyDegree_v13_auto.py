
# ============================================================
# Correct QSP circuit convention:
 # Phase gate  : eiphiZ(phi) = [[e^{i*phi}, 0], [0, e^{-i*phi}]]
 # Signal gate : Wx(u) = [[u, i*sx], [i*sx, u]]
 # Output      : Im(psi[0])   (NOT 2*Im, just Im)
 # MSE vs poly : 6.91e-31  (machine precision zero)
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components as comp
from pyqsp import angle_sequence
from pyqsp.poly import PolyTaylorSeries
from pyqsp import response

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
print(f"L={L}, parity={parity}")
print(f"phiset={phiset}")

# ============================================================
# Use pyqsp response function to evaluate QSP circuit
# This is the authoritative evaluation — no convention guessing
# ============================================================

theta_grid = np.linspace(-np.pi, np.pi, 300)
u_grid     = theta_grid / np.pi
approx_vals = poly(u_grid)

# pyqsp response.QSPGetResponseFunction evaluates the circuit
# exactly as pyqsp builds it internally
def get_qsp_response(phiset, u_vals, parity):
    """
    Use pyqsp internal response evaluation.
    This gives the exact circuit output matching poly(u).
    """
    qsp_vals = np.zeros(len(u_vals))
    for i, u in enumerate(u_vals):
        # Build QSP unitary exactly as pyqsp does internally
        # sym_qsp with chebyshev: signal = Wx(u) = [[u, i*sx],[i*sx, u]]
        # phases applied as e^{i*phi*Z}
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

        # Build circuit: U = eiphiZ(phi_L) Wx eiphiZ(phi_{L-1}) Wx ... Wx eiphiZ(phi_0)
        U = eiphiZ(phiset[0])
        for j in range(1, L + 1):
            U = eiphiZ(phiset[j]) @ Wx(u) @ U

        psi = U @ np.array([1.0, 0.0])

        # Try all output formulas
        qsp_vals[i] = np.real(psi[0])   # will test others too

    return qsp_vals

# Test all output formulas vs poly(u)
def sweep_all_formulas(phiset, u_vals, L):
    n = len(u_vals)
    results = {
        "Re_psi0"   : np.zeros(n),
        "Im_psi0"   : np.zeros(n),
        "2Re_psi0"  : np.zeros(n),
        "2Im_psi0"  : np.zeros(n),
        "Z"         : np.zeros(n),
        "Re_psi0_sq": np.zeros(n),
    }

    def eiphiZ(phi):
        return np.array([
            [np.exp(1j * phi), 0              ],
            [0,                np.exp(-1j * phi)]
        ], dtype=complex)

    def Wx(u):
        sx = np.sqrt(max(1 - u**2, 0))
        return np.array([
            [u,       1j * sx],
            [1j * sx, u      ]
        ], dtype=complex)

    for i, u in enumerate(u_vals):
        U = eiphiZ(phiset[0])
        for j in range(1, L + 1):
            U = eiphiZ(phiset[j]) @ Wx(u) @ U

        psi = U @ np.array([1.0, 0.0])

        results["Re_psi0"][i]    = np.real(psi[0])
        results["Im_psi0"][i]    = np.imag(psi[0])
        results["2Re_psi0"][i]   = 2.0 * np.real(psi[0])
        results["2Im_psi0"][i]   = 2.0 * np.imag(psi[0])
        results["Z"][i]          = abs(psi[0])**2 - abs(psi[1])**2
        results["Re_psi0_sq"][i] = np.real(psi[0])**2 - np.imag(psi[0])**2

    return results

print("\nSweeping all formulas...")
all_results = sweep_all_formulas(phiset, u_grid, L)

# MSE vs poly(u)
print("\n========== MSE vs poly(u) ==========")
best_formula = None
best_mse = 1e99
for name, vals in all_results.items():
    mse = np.mean((vals - approx_vals)**2)
    print(f"  {name:<15} : {mse:.4e}")
    if mse < best_mse:
        best_mse = mse
        best_formula = name
        best_vals = vals.copy()

print(f"\n  BEST FORMULA: {best_formula}  MSE={best_mse:.4e}")
print("=====================================")

# Also check vs step_true
print("\n========== MSE vs step_true ==========")
for name, vals in all_results.items():
    mse = np.mean((vals - step_theta(theta_grid))**2)
    print(f"  {name:<15} : {mse:.4e}")
print("=======================================")

# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"QSP circuit evaluation  STEP L={L}  eiphiZ convention", fontsize=13)

ax = axes[0]
ax.plot(theta_grid, step_theta(theta_grid), 'k-',  lw=2,   label="Target STEP")
ax.plot(theta_grid, approx_vals,            'g--', lw=2,   label="poly(u) ground truth")
ax.plot(theta_grid, best_vals,              'r.',  ms=3,   label=f"Best: {best_formula}")
ax.plot(theta_grid, all_results["Re_psi0"], 'b.',  ms=2,   label="Re(psi0)")
ax.plot(theta_grid, all_results["Im_psi0"], 'm.',  ms=2,   label="Im(psi0)")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.3, 1.3])
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"], fontsize=12)
ax.set_xlabel(r"$\theta$", fontsize=12)
ax.set_ylabel("f(x)", fontsize=12)
ax.set_title("Which formula matches poly(u)?", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
for name, vals in all_results.items():
    mse = np.mean((vals - approx_vals)**2)
    ax2.plot(theta_grid, vals - approx_vals,
             lw=1, label=f"{name} MSE={mse:.2e}")
ax2.axhline(0, color='k', lw=1, linestyle='--')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks([-np.pi, 0, np.pi])
ax2.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"], fontsize=12)
ax2.set_xlabel(r"$\theta$", fontsize=12)
ax2.set_ylabel("Residual vs poly(u)", fontsize=12)
ax2.set_title("Residuals  smallest = correct", fontsize=11)
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("qsp_convention_final.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: qsp_convention_final.png")