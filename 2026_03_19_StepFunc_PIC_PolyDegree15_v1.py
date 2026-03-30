import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components as comp

# ============================================================
# QSP phases from pyqsp, polydeg=15, STEP function
# ============================================================

phiset = np.array([
    -0.05263717,  0.05904224, -0.06758093,  0.07972595,
    -0.09871604,  0.13298893, -0.21285675,  0.56621395,
     0.56621395, -0.21285675,  0.13298893, -0.09871604,
     0.07972595, -0.06758093,  0.05904224, -0.05263717
])

L = 15
parity = 1
theta_all = np.pi / 2

print("=" * 55)
print("QSP Phase Configuration")
print("=" * 55)
print(f"  L (layers)     : {L}")
print(f"  Parity         : {parity}")
print(f"  Num phases     : {len(phiset)}")
print(f"  theta_j (all)  : {theta_all:.6f} rad")
print("=" * 55)


# ============================================================
# Build PIC MZI mesh in Perceval
# ============================================================

def build_qsp_pic(phiset, theta_all, x_val, L):
    circuit = pcvl.Circuit(2, name=f"QSP_STEP_L{L}")

    # Layer 0: no signal unitary
    circuit.add(0,      comp.PS(float(phiset[0])))
    circuit.add((0, 1), comp.BS.H(theta=float(theta_all) / 2))

    # Layers j = 1 to L
    for j in range(1, L + 1):
        circuit.add(0,      comp.PS(float(x_val)))
        circuit.add(0,      comp.PS(float(phiset[j])))
        circuit.add((0, 1), comp.BS.H(theta=float(theta_all) / 2))

    return circuit


# ============================================================
# Compute Z expectation from PIC unitary
# ============================================================

def get_z_expectation(phiset, theta_all, x_val, L):
    circuit = build_qsp_pic(phiset, theta_all, x_val, L)
    U = circuit.compute_unitary()
    p0 = abs(U[0, 0]) ** 2
    p1 = abs(U[1, 0]) ** 2
    return p0 - p1


# ============================================================
# Classical QSP simulation via matrix product
# ============================================================

def classical_qsp(phiset, theta_all, x_val, L):
    def Rz(phi):
        return np.array([
            [np.exp(-1j * phi / 2), 0],
            [0, np.exp(1j * phi / 2)]
        ])

    def Ry(theta):
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -s], [s, c]])

    W = Ry(theta_all) @ Rz(phiset[0])

    for j in range(1, L + 1):
        Rx_signal = Rz(x_val)
        A_j = Ry(theta_all) @ Rz(phiset[j])
        W = A_j @ Rx_signal @ W

    psi = W @ np.array([1.0, 0.0])
    p0 = abs(psi[0]) ** 2
    p1 = abs(psi[1]) ** 2
    return p0 - p1


# ============================================================
# Sweep x over [-pi, pi]
# ============================================================

n_points = 300
x_grid = np.linspace(-np.pi, np.pi, n_points)
f_pic = np.zeros(n_points)
f_classical = np.zeros(n_points)

print(f"\nSweeping x over [-pi, pi] with {n_points} points...")
for i, x in enumerate(x_grid):
    f_pic[i] = get_z_expectation(phiset, theta_all, x, L)
    f_classical[i] = classical_qsp(phiset, theta_all, x, L)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{n_points} done")

print("Sweep complete.")


# ============================================================
# MSE report
# ============================================================

step_true = np.where(x_grid >= 0, 1.0, -1.0)

mse_pic       = np.mean((f_pic       - step_true) ** 2)
mse_classical = np.mean((f_classical - step_true) ** 2)
mse_residual  = np.mean((f_pic - f_classical)     ** 2)

print("\n========== MSE Report ==========")
print(f"  MSE PIC vs target       : {mse_pic:.6f}")
print(f"  MSE Classical vs target : {mse_classical:.6f}")
print(f"  MSE PIC vs Classical    : {mse_residual:.8f}")
print("  If MSE PIC vs Classical is near zero,")
print("  the PIC correctly implements the QSP unitary.")
print("=================================")


# ============================================================
# PIC resource count
# ============================================================

circ_sample = build_qsp_pic(phiset, theta_all, 0.0, L)
n_ps = sum(1 for _, c in circ_sample._components
           if isinstance(c, comp.PS))
n_bs = sum(1 for _, c in circ_sample._components
           if isinstance(c, comp.BS))

print("\n========== PIC Resource Count ==========")
print(f"  QSP layers L        : {L}")
print(f"  Phase shifters PS   : {n_ps}")
print(f"  MZIs BS             : {n_bs}")
print(f"  Total components    : {n_ps + n_bs}")
print(f"  Waveguide modes     : 2 dual-rail qubit")
print("=========================================")


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Photonic QSP on PIC MZI mesh  STEP function L=15",
    fontsize=13
)

ax = axes[0]
ax.plot(x_grid, step_true,   'k-', lw=2,  label="Target STEP")
ax.plot(x_grid, f_classical, 'r-', lw=1.5, label="Classical QSP")
ax.plot(x_grid, f_pic,       'b.', ms=2,   label="Photonic PIC")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.3, 1.3])
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels(["-pi", "0", "pi"], fontsize=12)
ax.set_xlabel("Input signal x", fontsize=12)
ax.set_ylabel("Function value f(x)", fontsize=12)
ax.set_title(f"L = {L} layers", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
diff = f_pic - f_classical
ax2.plot(x_grid, diff, 'purple', lw=1.5, label="PIC minus Classical")
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.fill_between(x_grid, diff, alpha=0.2, color='purple')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks([-np.pi, 0, np.pi])
ax2.set_xticklabels(["-pi", "0", "pi"], fontsize=12)
ax2.set_xlabel("Input signal x", fontsize=12)
ax2.set_ylabel("Residual error", fontsize=12)
ax2.set_title("PIC vs Classical residual", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pic_qsp_step_L15.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: pic_qsp_step_L15.png")


# ============================================================
# Draw small circuit for paper (L=3)
# ============================================================

print("\nCircuit diagram for L=3 for paper:")
circ_small = build_qsp_pic(phiset[:4], theta_all, 0.5, L=3)
pcvl.pdisplay(circ_small)