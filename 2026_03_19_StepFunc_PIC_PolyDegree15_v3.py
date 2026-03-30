import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components as comp

phiset = np.array([
    -0.05263717,  0.05904224, -0.06758093,  0.07972595,
    -0.09871604,  0.13298893, -0.21285675,  0.56621395,
     0.56621395, -0.21285675,  0.13298893, -0.09871604,
     0.07972595, -0.06758093,  0.05904224, -0.05263717
])

L = 15
parity = 1

# ============================================================
# Matrix definitions
# ============================================================

def Rz_mat(phi):
    return np.array([
        [np.exp(1j * phi / 2), 0],
        [0, np.exp(-1j * phi / 2)]
    ])

def Rx_signal(x):
    sx = np.sqrt(max(1 - x**2, 0))
    return np.array([
        [x,       1j * sx],
        [1j * sx, x      ]
    ])

# ============================================================
# Classical QSP (verified correct - matches paper Fig.2)
# W = A_L * Rx * A_{L-1} * Rx * ... * Rx * A_0
# applied to |0> from the right
# ============================================================

def classical_qsp(phiset, u, L):
    W = Rz_mat(phiset[0])
    for j in range(1, L + 1):
        W = Rz_mat(phiset[j]) @ Rx_signal(u) @ W
    psi = W @ np.array([1.0, 0.0])
    return 2.0 * np.imag(psi[0])

# ============================================================
# Verify classical at single point
# ============================================================

u_test = 0.5
print(f"Classical at u=0.5: {classical_qsp(phiset, u_test, L):.6f}")

# ============================================================
# Manually compute PIC unitary as numpy matrix product
# matching EXACTLY the classical order, then verify
# ============================================================

def compute_pic_unitary_numpy(phiset, u_val, L):
    """
    Compute the full QSP unitary as a numpy matrix product.
    This is the ground truth for what the PIC should implement.
    Order matches classical_qsp exactly.
    """
    W = Rz_mat(phiset[0])
    for j in range(1, L + 1):
        W = Rz_mat(phiset[j]) @ Rx_signal(u_val) @ W
    return W

def get_numpy_output(phiset, u_val, L):
    W = compute_pic_unitary_numpy(phiset, u_val, L)
    psi = W @ np.array([1.0, 0.0])
    return 2.0 * np.imag(psi[0])

print(f"Numpy unitary at u=0.5: {get_numpy_output(phiset, u_test, L):.6f}")
print("(Should equal classical above)")

# ============================================================
# Build Perceval circuit matching numpy order exactly
#
# Classical builds W from RIGHT to LEFT:
#   W = Rz(L) @ Rx @ Rz(L-1) @ Rx @ ... @ Rx @ Rz(0)
#
# Perceval applies gates LEFT to RIGHT in circuit order.
# So we must add gates in REVERSE order:
#   first added = applied first to state = rightmost in W
#
# Circuit add order:
#   add Rz(phi_0)          [rightmost, applied first]
#   add Rx                 
#   add Rz(phi_1)          
#   ...
#   add Rx
#   add Rz(phi_L)          [leftmost, applied last]
# ============================================================

def build_qsp_pic(phiset, u_val, L):
    sx = np.sqrt(max(1 - u_val**2, 0))
    Rx = np.array([
        [u_val,   1j * sx],
        [1j * sx, u_val  ]
    ])

    circuit = pcvl.Circuit(2, name=f"QSP_STEP_L{L}")

    # Rz(phi_0) applied first (rightmost in W)
    circuit.add(0, comp.PS(float(phiset[0])))

    for j in range(1, L + 1):
        # Rx signal unitary
        circuit.add((0, 1), pcvl.Unitary(pcvl.Matrix(Rx)))
        # Rz(phi_j)
        circuit.add(0, comp.PS(float(phiset[j])))

    return circuit

def get_pic_output(phiset, u_val, L):
    circuit = build_qsp_pic(phiset, u_val, L)
    U = circuit.compute_unitary()
    psi_out = U @ np.array([1.0, 0.0])
    return 2.0 * np.imag(psi_out[0])

# ============================================================
# Verify Perceval PS convention matches Rz_mat
# PS(phi) in Perceval: check if it gives [[e^{i*phi/2}, 0], [0, e^{-i*phi/2}]]
# or [[e^{-i*phi/2}, 0], [0, e^{i*phi/2}]]
# ============================================================

print("\nVerifying Perceval PS convention:")
circ_ps = pcvl.Circuit(2)
circ_ps.add(0, comp.PS(np.pi))
U_ps = circ_ps.compute_unitary()
print(f"  PS(pi) = {U_ps[0,0]:.4f}, {U_ps[1,1]:.4f}")
print(f"  Our Rz(pi)[0,0] = {Rz_mat(np.pi)[0,0]:.4f}")
print(f"  Match: {np.isclose(U_ps[0,0], Rz_mat(np.pi)[0,0])}")

# ============================================================
# Sanity check: compare all three at u=0.5
# ============================================================

print("\n========== Sanity Check at u=0.5 ==========")
print(f"  Classical (verified) : {classical_qsp(phiset, u_test, L):.6f}")
print(f"  Numpy unitary        : {get_numpy_output(phiset, u_test, L):.6f}")
print(f"  Perceval PIC         : {get_pic_output(phiset, u_test, L):.6f}")
print("  All three must match.")
print("============================================")

# ============================================================
# If PS convention is flipped, use conjugate phi
# ============================================================

def get_pic_output_conj(phiset, u_val, L):
    """Try with negated phi angles in case PS convention is flipped."""
    sx = np.sqrt(max(1 - u_val**2, 0))
    Rx = np.array([
        [u_val,   1j * sx],
        [1j * sx, u_val  ]
    ])
    circuit = pcvl.Circuit(2, name=f"QSP_STEP_L{L}_conj")
    circuit.add(0, comp.PS(float(-phiset[0])))
    for j in range(1, L + 1):
        circuit.add((0, 1), pcvl.Unitary(pcvl.Matrix(Rx)))
        circuit.add(0, comp.PS(float(-phiset[j])))
    circuit_u = circuit.compute_unitary()
    psi_out = circuit_u @ np.array([1.0, 0.0])
    return 2.0 * np.imag(psi_out[0])

print(f"\n  Perceval PIC (negated phi) : {get_pic_output_conj(phiset, u_test, L):.6f}")
print("  If this matches classical, PS convention is flipped.")

# ============================================================
# Full sweep using whichever matches
# ============================================================

n_points = 300
u_grid = np.linspace(-1.0, 1.0, n_points)
x_grid = u_grid * np.pi

f_classical = np.zeros(n_points)
f_pic       = np.zeros(n_points)
f_pic_conj  = np.zeros(n_points)

print(f"\nSweeping {n_points} points...")
for i, u in enumerate(u_grid):
    f_classical[i] = classical_qsp(phiset, u, L)
    f_pic[i]       = get_pic_output(phiset, u, L)
    f_pic_conj[i]  = get_pic_output_conj(phiset, u, L)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{n_points} done")

print("Sweep complete.")

step_true = np.where(x_grid >= 0, 1.0, -1.0)

mse_pic      = np.mean((f_pic      - f_classical) ** 2)
mse_pic_conj = np.mean((f_pic_conj - f_classical) ** 2)

print(f"\n  MSE PIC vs Classical          : {mse_pic:.6f}")
print(f"  MSE PIC (negated) vs Classical: {mse_pic_conj:.6f}")
print("  Use whichever is closer to zero.")

# ============================================================
# Plot both versions to see which matches
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Photonic QSP STEP L=15 convention check", fontsize=13)

ax = axes[0]
ax.plot(x_grid, step_true,   'k-', lw=2,   label="Target STEP")
ax.plot(x_grid, f_classical, 'r-', lw=1.5, label="Classical QSP")
ax.plot(x_grid, f_pic,       'b.', ms=2,   label="PIC normal phi")
ax.plot(x_grid, f_pic_conj,  'g.', ms=2,   label="PIC negated phi")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.3, 1.3])
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels(["-pi", "0", "pi"], fontsize=12)
ax.set_xlabel("Input signal x", fontsize=12)
ax.set_ylabel("Function value f(x)", fontsize=12)
ax.set_title("Which PIC version matches classical?", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(x_grid, f_pic      - f_classical, 'b-', lw=1.5, label="PIC normal phi residual")
ax2.plot(x_grid, f_pic_conj - f_classical, 'g-', lw=1.5, label="PIC negated phi residual")
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks([-np.pi, 0, np.pi])
ax2.set_xticklabels(["-pi", "0", "pi"], fontsize=12)
ax2.set_xlabel("Input signal x", fontsize=12)
ax2.set_ylabel("Residual", fontsize=12)
ax2.set_title("Residuals: one should be near zero", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pic_qsp_convention_check.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: pic_qsp_convention_check.png")
