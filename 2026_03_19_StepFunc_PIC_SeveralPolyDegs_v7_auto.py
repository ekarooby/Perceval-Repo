import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components as comp
from pyqsp import angle_sequence
from pyqsp.poly import PolyTaylorSeries

# ============================================================
# SETTINGS: choose your polynomial degrees here
# ============================================================

poly_degrees = [15, 30, 60]   # add 180, 360 later (slow)
N_approx = 100

# ============================================================
# PART 1: Target functions
# ============================================================

def step_surrogate(u):
    return (2.0 / np.pi) * np.arctan(N_approx * np.pi * u)

def step_true(x):
    return np.where(x >= 0, 1.0, -1.0)

# ============================================================
# PART 2: Compute QSP phases for each degree
# ============================================================

def get_poly_and_phases(polydeg):
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
    print(f"  Phases : {phiset}")
    return poly, phiset, L

# ============================================================
# PART 3: Build Perceval PIC
# ============================================================

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
# PART 4: Sweep for each degree
# ============================================================

n_points = 200
u_grid = np.linspace(-1.0, 1.0, n_points)
x_grid = u_grid * np.pi

results = {}

for polydeg in poly_degrees:
    poly, phiset, L = get_poly_and_phases(polydeg)

    f_classical = np.zeros(n_points)
    f_pic       = np.zeros(n_points)

    print(f"  Sweeping {n_points} points for L={L}...")
    for i, u in enumerate(u_grid):
        f_classical[i] = float(poly(np.array([u]))[0])
        f_pic[i]       = get_pic_output(phiset, u, L)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n_points} done")

    step_vals     = step_true(x_grid)
    mse_classical = np.mean((f_classical - step_vals) ** 2)
    mse_pic       = np.mean((f_pic       - step_vals) ** 2)
    mse_residual  = np.mean((f_pic - f_classical)     ** 2)

    results[polydeg] = {
        "f_classical" : f_classical,
        "f_pic"       : f_pic,
        "mse_classic" : mse_classical,
        "mse_pic"     : mse_pic,
        "mse_residual": mse_residual,
        "L"           : L,
        "phiset"      : phiset,
    }

    print(f"  MSE poly vs Target  : {mse_classical:.6f}")
    print(f"  MSE PIC vs Target   : {mse_pic:.6f}")
    print(f"  MSE PIC vs poly     : {mse_residual:.2e}")

# ============================================================
# PART 5: MSE summary table (like paper Table I)
# ============================================================

print("\n========== MSE Summary Table ==========")
print(f"{'Degree':<10} {'L':<6} {'MSE poly':<15} {'MSE PIC':<15} {'MSE residual'}")
print("-" * 60)
for polydeg in poly_degrees:
    r = results[polydeg]
    print(f"{polydeg:<10} {r['L']:<6} {r['mse_classic']:<15.6f} "
          f"{r['mse_pic']:<15.6f} {r['mse_residual']:.2e}")
print("========================================")

# ============================================================
# PART 6: Plot — one row per degree, matching paper Fig.2
# ============================================================

n_deg = len(poly_degrees)
fig, axes = plt.subplots(n_deg, 2, figsize=(13, 5 * n_deg))
fig.suptitle("Photonic QSP on PIC  STEP function  multiple degrees", fontsize=14)

for row, polydeg in enumerate(poly_degrees):
    r = results[polydeg]
    L = r["L"]

    # Left panel: function comparison
    ax = axes[row, 0] if n_deg > 1 else axes[0]
    ax.plot(x_grid, step_true(x_grid), 'k-',  lw=2,   label="Target STEP")
    ax.plot(x_grid, r["f_classical"],  'r-',  lw=1.5, label="Classical poly(u)")
    ax.plot(x_grid, r["f_pic"],        'b.',  ms=2,   label="Photonic PIC")
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-1.3, 1.3])
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(["-pi", "0", "pi"], fontsize=11)
    ax.set_xlabel("Input signal x", fontsize=11)
    ax.set_ylabel("f(x)", fontsize=11)
    ax.set_title(f"polydeg={polydeg}  L={L}  MSE PIC={r['mse_pic']:.4f}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right panel: residual
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
    ax2.set_title(f"Residual PIC vs Classical  MSE={r['mse_residual']:.2e}", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pic_qsp_step_multidegree.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: pic_qsp_step_multidegree.png")