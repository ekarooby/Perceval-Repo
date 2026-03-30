
#With a detailed display, to avoid overlapping of labels
# ============================================================


import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components as comp

# ============================================================
# Load optimized angles
# ============================================================

theta_opt = np.load("theta_step_opt.npy")
phi_opt   = np.load("phi_step_opt.npy")
L         = len(theta_opt) - 1

print(f"Loaded QSP angles: L={L}")

N_approx = 100

def step_surrogate(x):
    return (2.0 / np.pi) * np.arctan(N_approx * x)

def step_true(x):
    return np.where(x >= 0, 1.0, -1.0)

# ============================================================
# Verified Perceval mappings:
#   Ry(theta) --> BS.Ry(theta)
#   Rz(phi)   --> PS(-phi/2) mode0 + PS(+phi/2) mode1
#   Output    --> Z = p0-p1
# ============================================================

def Ry_mat(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s],[s, c]], dtype=complex)

def Rz_mat(phi):
    return np.array([
        [np.exp(-1j*phi/2), 0              ],
        [0,                 np.exp(1j*phi/2)]
    ], dtype=complex)

def A_mat(theta, phi):
    return Ry_mat(theta) @ Rz_mat(phi)

def classical_qsp(theta_arr, phi_arr, x_val, L):
    W = A_mat(theta_arr[0], phi_arr[0])
    for j in range(1, L + 1):
        W = A_mat(theta_arr[j], phi_arr[j]) @ Rz_mat(x_val) @ W
    psi = W @ np.array([1.0, 0.0])
    return abs(psi[0])**2 - abs(psi[1])**2

def build_qsp_pic(theta_arr, phi_arr, x_val, L):
    circuit = pcvl.Circuit(2, name=f"QSP_STEP_L{L}")

    # A(theta_0, phi_0) = Ry(theta_0) * Rz(phi_0)
    # Perceval applies LEFT to RIGHT so add Rz first then Ry
    circuit.add(0,      comp.PS(float(-phi_arr[0] / 2)))   # Rz first
    circuit.add(1,      comp.PS(float( phi_arr[0] / 2)))
    circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[0])))  # then Ry

    for j in range(1, L + 1):
        # Rz(x): signal unitary
        circuit.add(0, comp.PS(float(-x_val / 2)))
        circuit.add(1, comp.PS(float( x_val / 2)))
        # A(theta_j, phi_j) = Ry * Rz: add Rz first then Ry
        circuit.add(0,      comp.PS(float(-phi_arr[j] / 2)))
        circuit.add(1,      comp.PS(float( phi_arr[j] / 2)))
        circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[j])))

    return circuit

def get_pic_z(theta_arr, phi_arr, x_val, L):
    circuit = build_qsp_pic(theta_arr, phi_arr, x_val, L)
    U       = np.array(circuit.compute_unitary())
    psi     = U @ np.array([1.0, 0.0])
    return abs(psi[0])**2 - abs(psi[1])**2

# ============================================================
# Sanity check
# ============================================================

print("\n========== Sanity Check ==========")
for x_test in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    c_val = classical_qsp(theta_opt, phi_opt, x_test, L)
    p_val = get_pic_z(theta_opt, phi_opt, x_test, L)
    f_ref = step_surrogate(x_test)
    print(f"  x={x_test:+.1f}: surrogate={f_ref:.4f}  "
          f"classical={c_val:.4f}  PIC={p_val:.4f}  "
          f"diff={abs(c_val-p_val):.2e}")
print("===================================")

# ============================================================
# Full sweep
# ============================================================

theta_grid  = np.linspace(-np.pi, np.pi, 300)
f_classical = np.zeros(len(theta_grid))
f_pic       = np.zeros(len(theta_grid))
f_surrogate = np.array([step_surrogate(x) for x in theta_grid])
f_true      = step_true(theta_grid)

print(f"\nSweeping {len(theta_grid)} points...")
for i, x in enumerate(theta_grid):
    f_classical[i] = classical_qsp(theta_opt, phi_opt, x, L)
    f_pic[i]       = get_pic_z(theta_opt, phi_opt, x, L)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(theta_grid)} done")

print("Sweep complete.")

# ============================================================
# MSE report
# ============================================================

mse_pic_classic = np.mean((f_pic - f_classical) ** 2)
mse_pic_surr    = np.mean((f_pic - f_surrogate) ** 2)
mse_pic_true    = np.mean((f_pic - f_true)      ** 2)

print("\n========== MSE Report ==========")
print(f"  MSE PIC vs Classical : {mse_pic_classic:.2e}  must be ~0")
print(f"  MSE PIC vs surrogate : {mse_pic_surr:.4f}")
print(f"  MSE PIC vs true STEP : {mse_pic_true:.4f}")
print(f"  Output Z=p0-p1 is directly measurable with detectors")
print("=================================")

# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Photonic QSP on PIC  STEP L={L}  Z=p0-p1  measurable",
    fontsize=13
)

ax = axes[0]
ax.plot(theta_grid, f_true,      'k-',  lw=2,   label="True STEP")
ax.plot(theta_grid, f_surrogate, 'g--', lw=2,   label="arctan surrogate")
ax.plot(theta_grid, f_classical, 'r-',  lw=1.5, label="Classical Z=p0-p1")
ax.plot(theta_grid, f_pic,       'b.',  ms=3,
        label=f"Perceval analysis Z=p0-p1  Perceval analysis minus STEP MSE={mse_pic_true:.4f}")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.3, 1.3])
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"], fontsize=12)
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel("Function value", fontsize=12)
ax.set_title(f"L={L}  blue must sit on red", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
diff = f_pic - f_classical
ax2.plot(theta_grid, diff, 'purple', lw=1.5,
         label=f"Perceval analysis minus Classical  MSE={mse_pic_classic:.2e}")
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.fill_between(theta_grid, diff, alpha=0.2, color='purple')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks([-np.pi, 0, np.pi])
ax2.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"], fontsize=12)
ax2.set_xlabel(r"$x$", fontsize=12)
ax2.set_ylabel("Residual", fontsize=12)
ax2.set_title("PIC vs Classical  must be ~0", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pic_qsp_step_measurable_FINAL.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: pic_qsp_step_measurable_FINAL.png")

# ============================================================
# PIC resource count
# ============================================================

circ_full = build_qsp_pic(theta_opt, phi_opt, 0.5, L)
n_ps = sum(1 for _, c in circ_full._components if isinstance(c, comp.PS))
n_bs = sum(1 for _, c in circ_full._components if isinstance(c, comp.BS))

print("\n========== PIC Resource Count for Quandela ==========")
print(f"  QSP layers L        : {L}")
print(f"  Phase shifters PS   : {n_ps}")
print(f"  Beam splitters BS   : {n_bs}")
print(f"  Total components    : {n_ps + n_bs}")
print(f"  Waveguide modes     : 2  dual-rail qubit")
print(f"  Output measurement  : Z = p0-p1 photon counting")
print("=====================================================")

# ============================================================
# Draw QSP circuit manually - full control over layout
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

def draw_qsp_circuit(theta_arr, phi_arr, L_draw=3, x_val=0.5):
    """
    Draw QSP circuit manually for L_draw layers.
    Structure per layer:
      Rz(x): PS(-x/2) mode0, PS(+x/2) mode1
      A(t,p): PS(-p/2) mode0, PS(+p/2) mode1, BS.Ry(t)
    """

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')

    # Wire y positions
    y0, y1 = 2.0, 0.5   # mode 0 top, mode 1 bottom

    # ---- helper functions ----
    def ps_box(ax, x_center, y, label, color='#4C72B0', width=0.045, height=0.35):
        box = FancyBboxPatch(
            (x_center - width/2, y - height/2),
            width, height,
            boxstyle="round,pad=0.005",
            facecolor=color, edgecolor='black', linewidth=0.8, zorder=3
        )
        ax.add_patch(box)
        ax.text(x_center, y, label, ha='center', va='center',
                fontsize=6.5, color='white', fontweight='bold', zorder=4)

    def bs_box(ax, x_center, y0, y1, label):
        yc   = (y0 + y1) / 2
        half = (y0 - y1) / 2 + 0.1
        box  = FancyBboxPatch(
            (x_center - 0.055, yc - half),
            0.11, 2 * half,
            boxstyle="round,pad=0.005",
            facecolor='#C44E52', edgecolor='black', linewidth=0.8, zorder=3
        )
        ax.add_patch(box)
        ax.text(x_center, yc, label, ha='center', va='center',
                fontsize=6.5, color='white', fontweight='bold', zorder=4)

    def wire(ax, x_start, x_end, y):
        ax.plot([x_start, x_end], [y, y], 'k-', lw=1.2, zorder=1)

    # ---- layout ----
    # x positions for each block
    # Initial A(t0,p0): at x~0.04
    # Then L_draw repetitions of [Rz(x), A(tj,pj)]

    n_blocks  = 1 + L_draw          # initial + L layers
    block_w   = 0.85 / n_blocks      # width per block
    x_start   = 0.07

    # Draw full wires first
    wire(ax, 0.01, 0.99, y0)
    wire(ax, 0.01, 0.99, y1)

    # Mode labels
    ax.text(0.01, y0 + 0.2, 'mode 0', fontsize=8, va='bottom', ha='left')
    ax.text(0.01, y1 - 0.2, 'mode 1', fontsize=8, va='top',    ha='left')

    # ---- Initial block A(θ₀, φ₀) ----
    xc = x_start + block_w * 0.3

    ps_label0 = f"PS\n{-phi_arr[0]/2:.3f}"
    ps_label1 = f"PS\n{+phi_arr[0]/2:.3f}"
    bs_label  = f"BS.Ry\nθ={theta_arr[0]:.3f}"

    ps_box(ax, xc,            y0, ps_label0)
    ps_box(ax, xc,            y1, ps_label1, color='#4C72B0')
    bs_box(ax, xc + block_w * 0.45, y0, y1, bs_label)

    ax.text(xc - 0.01, y0 + 0.45, f'A(θ₀,φ₀)', fontsize=7,
            ha='center', color='gray', style='italic')

    # ---- Layers 1..L_draw ----
    for j in range(1, L_draw + 1):
        x_block = x_start + j * block_w
        xc_rz   = x_block + block_w * 0.15
        xc_ps   = x_block + block_w * 0.45
        xc_bs   = x_block + block_w * 0.75

        # Rz(x) signal unitary
        ps_box(ax, xc_rz, y0, f"PS\n{-x_val/2:.3f}", color='#55A868')
        ps_box(ax, xc_rz, y1, f"PS\n{+x_val/2:.3f}", color='#55A868')
        ax.text(xc_rz, y0 + 0.45, f'Rz(x)', fontsize=7,
                ha='center', color='#55A868', style='italic')

        # A(θⱼ, φⱼ)
        ps_box(ax, xc_ps, y0, f"PS\n{-phi_arr[j]/2:.3f}")
        ps_box(ax, xc_ps, y1, f"PS\n{+phi_arr[j]/2:.3f}")
        bs_box(ax, xc_bs, y0, y1, f"BS.Ry\nθ={theta_arr[j]:.3f}")
        ax.text(xc_ps + (xc_bs - xc_ps)/2, y0 + 0.45,
                f'A(θ{j},φ{j})', fontsize=7,
                ha='center', color='gray', style='italic')

    # ---- Detectors ----
    for y, lbl in [(y0, 'D₀\n(p₀)'), (y1, 'D₁\n(p₁)')]:
        box = FancyBboxPatch((0.94, y - 0.18), 0.05, 0.36,
                             boxstyle="round,pad=0.005",
                             facecolor='#8172B2', edgecolor='black',
                             linewidth=0.8, zorder=3)
        ax.add_patch(box)
        ax.text(0.965, y, lbl, ha='center', va='center',
                fontsize=6.5, color='white', fontweight='bold', zorder=4)

    ax.text(0.965, y1 - 0.42, 'Z = p₀ - p₁', fontsize=7.5,
            ha='center', color='purple', fontweight='bold')

    ax.set_title(
        f'Photonic QSP Circuit  —  L={L_draw} layers shown  '
        f'(full circuit L=15)  |  Blue=PS(φ)  Red=BS.Ry(θ)  Green=PS(x signal)  Purple=Detector',
        fontsize=9, pad=10
    )

    plt.tight_layout()
    plt.savefig("pic_qsp_circuit_manual.png", dpi=200, bbox_inches='tight')
    plt.show()
    print("Saved: pic_qsp_circuit_manual.png")

# Run it
draw_qsp_circuit(theta_opt, phi_opt, L_draw=3, x_val=0.5) # This drawas onlt 3 layers