import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components as comp
from pyqsp import angle_sequence
from pyqsp.poly import PolyTaylorSeries

N_approx = 100
poly_degrees = [15, 30, 60]

def step_surrogate(u):
    return (2.0 / np.pi) * np.arctan(N_approx * np.pi * u)

def step_true(x):
    return np.where(x >= 0, 1.0, -1.0)

# ============================================================
# Perceval PS convention (VERIFIED in v4):
# PS(phi) = [[e^{i*phi}, 0],[0, 1]]
# Output  = 2*Im(psi[0])  --> MSE=0 confirmed in v4
# ============================================================

def PS_mat(phi):
    return np.array([
        [np.exp(1j * phi), 0],
        [0,                1]
    ], dtype=complex)

def Rx_signal(x):
    sx = np.sqrt(max(1 - x**2, 0))
    return np.array([
        [x,       1j * sx],
        [1j * sx, x      ]
    ], dtype=complex)

def classical_qsp_ps(phiset, u, L):
    """
    Classical QSP using VERIFIED Perceval PS convention.
    This matched PIC exactly (MSE=0) in v4.
    """
    W = PS_mat(phiset[0])
    for j in range(1, L + 1):
        W = PS_mat(phiset[j]) @ Rx_signal(u) @ W
    psi = W @ np.array([1.0, 0.0])
    return 2.0 * np.imag(psi[0])

def build_qsp_pic(phiset, u_val, L):
    sx = np.sqrt(max(1 - u_val**2, 0))
    Rx = np.array([
        [u_val,   1j * sx],
        [1j * sx, u_val  ]
    ], dtype=complex)
    circuit = pcvl.Circuit(2, name=f"QSP_L{L}")
    circuit.add(0, comp.PS(float(phiset[0])))
    for j in range(1, L + 1):
        circuit.add((0, 1), pcvl.Unitary(pcvl.Matrix(Rx)))
        circuit.add(0, comp.PS(float(phiset[j])))
    return circuit

def get_pic_output(phiset, u_val, L):
    circuit = build_qsp_pic(phiset, u_val, L)
    U = circuit.compute_unitary()
    psi_out = U @ np.array([1.0, 0.0])
    return 2.0 * np.imag(psi_out[0])

# ============================================================
# Get phases and sweep for each degree
# ============================================================

n_points = 200
u_grid = np.linspace(-1.0, 1.0, n_points)
x_grid = u_grid * np.pi
results = {}

for polydeg in poly_degrees:
    print(f"\n--- Computing phases for polydeg={polydeg} ---")

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

    # Sanity check at u=0.5
    c_val = classical_qsp_ps(phiset, 0.5, L)
    p_val = get_pic_output(phiset, 0.5, L)
    print(f"  Sanity check u=0.5:")
    print(f"    Classical PS : {c_val:.6f}")
    print(f"    PIC          : {p_val:.6f}")
    print(f"    Difference   : {abs(c_val-p_val):.2e}  (must be ~0)")

    f_classical = np.zeros(n_points)
    f_pic       = np.zeros(n_points)

    print(f"  Sweeping {n_points} points...")
    for i, u in enumerate(u_grid):
        f_classical[i] = classical_qsp_ps(phiset, u, L)
        f_pic[i]       = get_pic_output(phiset, u, L)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n_points} done")

    step_vals     = step_true(x_grid)
    mse_classical = np.mean((f_classical - step_vals) ** 2)
    mse_pic       = np.mean((f_pic       - step_vals) ** 2)
    mse_residual  = np.mean((f_pic - f_classical)     ** 2)

    print(f"  MSE Classical vs Target : {mse_classical:.6f}")
    print(f"  MSE PIC vs Target       : {mse_pic:.6f}")
    print(f"  MSE PIC vs Classical    : {mse_residual:.2e}")

    results[polydeg] = {
        "f_classical"  : f_classical,
        "f_pic"        : f_pic,
        "mse_classic"  : mse_classical,
        "mse_pic"      : mse_pic,
        "mse_residual" : mse_residual,
        "L"            : L,
    }

# ============================================================
# MSE summary table
# ============================================================

print("\n========== MSE Summary Table ==========")
print(f"{'Degree':<10} {'L':<6} {'MSE Classical':<16} {'MSE PIC':<16} {'MSE residual'}")
print("-" * 65)
for polydeg in poly_degrees:
    r = results[polydeg]
    print(f"{polydeg:<10} {r['L']:<6} {r['mse_classic']:<16.6f} "
          f"{r['mse_pic']:<16.6f} {r['mse_residual']:.2e}")
print("========================================")

# ============================================================
# Plot
# ============================================================

n_deg = len(poly_degrees)
fig, axes = plt.subplots(n_deg, 2, figsize=(13, 5 * n_deg))
fig.suptitle("Photonic QSP on PIC  STEP function  multiple degrees", fontsize=14)

for row, polydeg in enumerate(poly_degrees):
    r = results[polydeg]
    L = r["L"]

    ax = axes[row, 0] if n_deg > 1 else axes[0]
    ax.plot(x_grid, step_true(x_grid), 'k-',  lw=2,   label="Target STEP")
    ax.plot(x_grid, r["f_classical"],  'r-',  lw=1.5, label="Classical QSP")
    ax.plot(x_grid, r["f_pic"],        'b.',  ms=2,   label="Photonic PIC")
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-1.3, 1.3])
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(["-pi", "0", "pi"], fontsize=11)
    ax.set_xlabel("Input signal x", fontsize=11)
    ax.set_ylabel("f(x)", fontsize=11)
    ax.set_title(
        f"polydeg={polydeg}  L={L}  "
        f"MSE Classic={r['mse_classic']:.4f}  "
        f"MSE PIC={r['mse_pic']:.4f}",
        fontsize=10
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[row, 1] if n_deg > 1 else axes[1]
    diff = r["f_pic"] - r["f_classical"]
    ax2.plot(x_grid, diff, 'purple', lw=1.5, label="PIC minus Classical")
    ax2.axhline(0, color='k', lw=0.8, linestyle='--')
    ax2.fill_between(x_grid, diff, alpha=0.2, color='purple')
    ax2.set_xlim([-np.pi, np.pi])
    ax2.set_xticks([-np.pi, 0, np.pi])
    ax2.set_xticklabels(["-pi", "0", "pi"], fontsize=11)
    ax2.set_xlabel("Input signal x", fontsize=11)
    ax2.set_ylabel("Residual", fontsize=11)
    ax2.set_title(
        f"PIC vs Classical residual  MSE={r['mse_residual']:.2e}",
        fontsize=11
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pic_qsp_step_multidegree_FINAL.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: pic_qsp_step_multidegree_FINAL.png")