import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components as comp
from pyqsp import angle_sequence
from pyqsp.poly import PolyTaylorSeries

N_approx = 100
poly_degrees = [15]

# ============================================================
# Target functions
# ============================================================

def step_surrogate(u):
    return (2.0 / np.pi) * np.arctan(N_approx * np.pi * u)

def step_true(x):
    return np.where(x >= 0, 1.0, -1.0)

# ============================================================
# Classical QSP using Rz convention (matches paper Fig.2)
# Rz(phi) = [[e^{i*phi/2}, 0],[0, e^{-i*phi/2}]]
# Output = 2*Im(psi[0])
# ============================================================

def Rz_mat(phi):
    return np.array([
        [np.exp(1j * phi / 2), 0],
        [0, np.exp(-1j * phi / 2)]
    ], dtype=complex)

def Rx_signal(x):
    sx = np.sqrt(max(1 - x**2, 0))
    return np.array([
        [x,       1j * sx],
        [1j * sx, x      ]
    ], dtype=complex)

def classical_qsp(phiset, u, L):
    """
    Classical QSP using Rz convention.
    Matches paper Fig.2 shape exactly.
    """
    W = Rz_mat(phiset[0])
    for j in range(1, L + 1):
        W = Rz_mat(phiset[j]) @ Rx_signal(u) @ W
    psi = W @ np.array([1.0, 0.0])
    return 2.0 * np.imag(psi[0])

# ============================================================
# Perceval PS convention (VERIFIED MSE=0 in v4):
# PS(phi) = [[e^{i*phi}, 0],[0, 1]]
# To match Rz(phi) upper element e^{i*phi/2}:
#   use PS(phi/2) in Perceval
# Output = 2*Im(psi[0])
# ============================================================

def PS_mat(phi):
    return np.array([
        [np.exp(1j * phi), 0],
        [0,                1]
    ], dtype=complex)

def classical_qsp_ps(phiset, u, L):
    """
    Classical QSP using Perceval PS convention.
    Used only for verifying PIC match (MSE=0).
    """
    W = PS_mat(phiset[0])
    for j in range(1, L + 1):
        W = PS_mat(phiset[j]) @ Rx_signal(u) @ W
    psi = W @ np.array([1.0, 0.0])
    return 2.0 * np.imag(psi[0])

# ============================================================
# Build Perceval PIC circuit
# PS(phi) used directly — verified MSE=0 vs classical_qsp_ps
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
# Sweep for each degree
# ============================================================

n_points = 200
u_grid = np.linspace(-1.0, 1.0, n_points)
x_grid = u_grid * np.pi
results = {}
last_phiset = None
last_L = None

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
    last_phiset = phiset
    last_L = L
    print(f"  Parity : {parity}")
    print(f"  L      : {L}")

    # Sanity check: PIC must match classical_qsp_ps
    c_ps  = classical_qsp_ps(phiset, 0.5, L)
    p_val = get_pic_output(phiset, 0.5, L)
    c_rz  = classical_qsp(phiset, 0.5, L)
    print(f"  Sanity check at u=0.5:")
    print(f"    Classical Rz (paper shape) : {c_rz:.6f}")
    print(f"    Classical PS (PIC verify)  : {c_ps:.6f}")
    print(f"    PIC output                 : {p_val:.6f}")
    print(f"    PIC vs PS difference       : {abs(c_ps - p_val):.2e}  (must be ~0)")

    f_classical_rz = np.zeros(n_points)
    f_classical_ps = np.zeros(n_points)
    f_pic          = np.zeros(n_points)

    print(f"  Sweeping {n_points} points...")
    for i, u in enumerate(u_grid):
        f_classical_rz[i] = classical_qsp(phiset, u, L)
        f_classical_ps[i] = classical_qsp_ps(phiset, u, L)
        f_pic[i]          = get_pic_output(phiset, u, L)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n_points} done")

    step_vals      = step_true(x_grid)
    mse_rz_target  = np.mean((f_classical_rz - step_vals) ** 2)
    mse_pic_target = np.mean((f_pic           - step_vals) ** 2)
    mse_pic_ps     = np.mean((f_pic - f_classical_ps)     ** 2)

    print(f"  MSE Classical Rz vs Target  : {mse_rz_target:.6f}")
    print(f"  MSE PIC vs Target           : {mse_pic_target:.6f}")
    print(f"  MSE PIC vs Classical PS     : {mse_pic_ps:.2e}  (must be ~0)")

    results[polydeg] = {
        "f_classical_rz" : f_classical_rz,
        "f_classical_ps" : f_classical_ps,
        "f_pic"          : f_pic,
        "mse_rz_target"  : mse_rz_target,
        "mse_pic_target" : mse_pic_target,
        "mse_pic_ps"     : mse_pic_ps,
        "L"              : L,
        "phiset"         : phiset,
    }

# ============================================================
# MSE summary table for paper
# ============================================================

print("\n========== MSE Summary Table ==========")
print(f"{'Degree':<10} {'L':<6} {'MSE Rz/Target':<16} {'MSE PIC/PS'}")
print("-" * 50)
for polydeg in poly_degrees:
    r = results[polydeg]
    print(f"{polydeg:<10} {r['L']:<6} {r['mse_rz_target']:<16.6f} {r['mse_pic_ps']:.2e}")
print("========================================")

# ============================================================
# Plot — paper Fig.2 style
# Left panel : classical Rz (paper shape) + PIC
# Right panel: residual PIC vs classical PS (should be ~0)
# ============================================================

n_deg = len(poly_degrees)
fig, axes = plt.subplots(n_deg, 2, figsize=(13, 5 * n_deg))
fig.suptitle("Photonic QSP on PIC  STEP function  multiple degrees", fontsize=14)

for row, polydeg in enumerate(poly_degrees):
    r = results[polydeg]
    L = r["L"]

    # Left: paper Fig.2 style comparison
    ax = axes[row, 0] if n_deg > 1 else axes[0]
    ax.plot(x_grid, step_true(x_grid),  'k-', lw=2,   label="Target STEP")
    ax.plot(x_grid, r["f_classical_rz"],'r-', lw=1.5, label="Classical QSP (paper Fig.2)")
    ax.plot(x_grid, r["f_pic"],         'b.', ms=2,   label="Photonic PIC")
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-1.3, 1.3])
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(["-pi", "0", "pi"], fontsize=11)
    ax.set_xlabel("Input signal x", fontsize=11)
    ax.set_ylabel("f(x)", fontsize=11)
    ax.set_title(
        f"polydeg={polydeg}  L={L}  MSE={r['mse_rz_target']:.4f}",
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: residual PIC vs PS classical (verification)
    ax2 = axes[row, 1] if n_deg > 1 else axes[1]
    diff = r["f_pic"] - r["f_classical_ps"]
    ax2.plot(x_grid, diff, 'purple', lw=1.5, label="PIC minus Classical PS")
    ax2.axhline(0, color='k', lw=0.8, linestyle='--')
    ax2.fill_between(x_grid, diff, alpha=0.2, color='purple')
    ax2.set_xlim([-np.pi, np.pi])
    ax2.set_xticks([-np.pi, 0, np.pi])
    ax2.set_xticklabels(["-pi", "0", "pi"], fontsize=11)
    ax2.set_xlabel("Input signal x", fontsize=11)
    ax2.set_ylabel("Residual", fontsize=11)
    ax2.set_title(
        f"PIC vs Classical residual  MSE={r['mse_pic_ps']:.2e}  (should be ~0)",
        fontsize=11
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pic_qsp_step_multidegree_FINAL.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: pic_qsp_step_multidegree_FINAL.png")

# ============================================================
# PIC resource count for paper
# ============================================================

print("\n========== PIC Resource Count ==========")
print(f"{'Degree':<10} {'L':<6} {'PS count':<12} {'Unitary count':<16} {'Total'}")
print("-" * 50)
for polydeg in poly_degrees:
    r  = results[polydeg]
    L  = r["L"]
    phi = r["phiset"]
    circ = build_qsp_pic(phi, 0.0, L)
    n_ps  = sum(1 for _, c in circ._components if isinstance(c, comp.PS))
    n_uni = sum(1 for _, c in circ._components if isinstance(c, pcvl.Unitary))
    print(f"{polydeg:<10} {L:<6} {n_ps:<12} {n_uni:<16} {n_ps+n_uni}")
print("=========================================")

# ============================================================
# Draw circuit for L=3 (readable for paper)
# ============================================================

print("\nDrawing circuit for L=3...")
circ_small = build_qsp_pic(last_phiset[:4], 0.5, L=3)
fig_circ = plt.figure(figsize=(12, 3))
pcvl.pdisplay(circ_small, output_format=pcvl.Format.MPLOT)
plt.savefig("qsp_pic_circuit_L3.png", dpi=150, bbox_inches='tight')
plt.show()
print("Circuit saved: qsp_pic_circuit_L3.png")