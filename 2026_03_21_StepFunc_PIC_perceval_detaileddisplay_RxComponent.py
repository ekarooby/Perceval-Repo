# ============================================================
# PHOTONIC QSP ON PIC - HARDWARE COMPATIBLE VERSION v2
# Date: 2026-03-21
# ============================================================
#
# GOAL:
#   Implement Quantum Signal Processing (QSP) on a Quandela
#   Photonic Integrated Circuit (PIC) to approximate a STEP
#   function, using only hardware-native components.
#
# WHAT IS QSP?
#   QSP is a quantum algorithm framework that approximates
#   a target function f(x) by applying a sequence of signal
#   unitaries Rz(x) interleaved with tunable rotation gates
#   A(theta, phi). The circuit structure is:
#
#   W(x) = A(t0,p0) * [Rz(x) * A(t1,p1)] * ... * [Rz(x) * A(tL,pL)]
#
#   where A(theta,phi) = Ry(theta) * Rz(phi)
#   and x is the input signal swept over [-pi, pi].
#   The output is Z = p0 - p1 (difference of photon counts).
#
# WHY STEP FUNCTION?
#   The STEP function is a fundamental target in QSP.
#   It is approximated using an arctan surrogate:
#     f(x) = (2/pi) * arctan(100*x)
#   which closely mimics a step at x=0. We use L=15 layers.
#
# HOW WERE THE ANGLES OPTIMIZED?
#   theta_opt and phi_opt were found by scipy optimization,
#   minimizing MSE between Z=p0-p1 and the arctan surrogate
#   over a grid of x values. Results are loaded from .npy files.
#   MSE vs true STEP achieved: ~0.0405 (comparable to paper).
#
# THE HARDWARE CHALLENGE:
#   The QSP circuit uses Ry(theta) rotations with variable theta.
#   Quandela hardware only has:
#     - Fixed 50:50 beam splitters (no variable splitting ratio)
#     - Programmable phase shifters (PS)
#   So BS.Ry(theta) cannot be directly implemented on hardware.
#
# THE SOLUTION - BS.Rx DECOMPOSITION:
#   We decompose BS.Ry(theta) into hardware-native components:
#
#     BS.Ry(theta) = Rz(-pi/2) @ BS.Rx(theta) @ Rz(+pi/2)
#
#   where:
#     BS.Rx(theta) = primitive Perceval gate, NOT a true MZI.
#                   It is defined directly by its matrix:
#                   [[cos(t/2), i*sin(t/2)],
#                    [i*sin(t/2), cos(t/2)]]
#                   On Quandela hardware it is physically realized
#                   as: BS.Rx(pi/2) + PS(phi) + BS.Rx(pi/2)
#                   but Perceval abstracts this away as a single
#                   BS.Rx component. We use it as a primitive here.
#     Rz(phi)      = PS(-phi/2) on mode0 + PS(+phi/2) on mode1
#
#   This was found by:
#   1. Testing BS.H + PS + BS.H  --> FAILED (diff=0.84, wrong convention)
#   2. Testing all BS conventions --> FAILED (none matched directly)
#   3. Using identity: Ry = Rz(-pi/2) @ Rx @ Rz(+pi/2) --> CORRECT
#
# PERCEVAL CONVENTIONS (critical):
#   - Gates applied LEFT TO RIGHT in Perceval
#   - Rz(phi)      --> PS(-phi/2) mode0 + PS(+phi/2) mode1
#   - BS.Rx(theta) is a primitive Perceval gate, NOT a true MZI
#   - BS.Ry(theta) exists in simulation only, not on hardware
#
# OUTPUT MEASUREMENT:
#   Z = p0 - p1 = |psi[0]|^2 - |psi[1]|^2
#   Directly measurable with single photon detectors.
#   No phase-sensitive measurement needed.
#
# HARDWARE RESOURCE COUNT (L=15):
#   BS.Rx (primitive gate) : 16
#   Phase shifters PS      : 126
#   Total                  : 142
#   Waveguide modes        : 2 (dual-rail qubit)
#
# RESULTS:
#   MSE hardware vs classical : 6.57e-30 (machine precision zero)
#   MSE hardware vs true STEP : 0.0405
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import perceval as pcvl
import perceval.components as comp

# ============================================================
# Load optimized QSP angles from scipy optimization
# theta_opt : rotation angles for BS.Rx gates  (L+1 values)
# phi_opt   : rotation angles for Rz gates     (L+1 values)
# ============================================================

theta_opt = np.load("theta_step_opt.npy")
phi_opt   = np.load("phi_step_opt.npy")
L         = len(theta_opt) - 1
print(f"Loaded QSP angles: L={L}")

N_approx = 100

def step_surrogate(x):
    """Smooth approximation of STEP used during optimization (paper Eq. B9)."""
    return (2.0 / np.pi) * np.arctan(N_approx * x)

def step_true(x):
    """Ideal STEP function: -1 for x<0, +1 for x>=0."""
    return np.where(x >= 0, 1.0, -1.0)

# ============================================================
# Classical (matrix) QSP reference
# Used to verify the PIC circuit gives identical results.
# ============================================================

def Ry_mat(theta):
    """2x2 real rotation matrix Ry(theta)."""
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -s],[s, c]], dtype=complex)

def Rz_mat(phi):
    """2x2 phase rotation matrix Rz(phi)."""
    return np.array([[np.exp(-1j*phi/2), 0],
                     [0, np.exp(1j*phi/2)]], dtype=complex)

def A_mat(theta, phi):
    """QSP building block: A(theta,phi) = Ry(theta) @ Rz(phi)."""
    return Ry_mat(theta) @ Rz_mat(phi)

def classical_qsp(theta_arr, phi_arr, x_val, L):
    """
    Classical matrix computation of QSP circuit output Z = p0-p1.
    W(x) = A(t0,p0) * prod_{j=1}^{L} [A(tj,pj) @ Rz(x)]
    Applied LEFT to RIGHT (matching Perceval convention).
    """
    W = A_mat(theta_arr[0], phi_arr[0])
    for j in range(1, L + 1):
        W = A_mat(theta_arr[j], phi_arr[j]) @ Rz_mat(x_val) @ W
    psi = W @ np.array([1.0, 0.0])
    return abs(psi[0])**2 - abs(psi[1])**2

# ============================================================
# Hardware-native circuit building blocks
#
# add_rz: implements Rz(phi) using two PS gates
#   PS(-phi/2) on mode 0
#   PS(+phi/2) on mode 1
#
# add_ry_as_bsrx: implements BS.Ry(theta) using hardware-native
#   components only, via the identity:
#   BS.Ry(theta) = Rz(-pi/2) @ BS.Rx(theta) @ Rz(+pi/2)
#   In Perceval (LEFT TO RIGHT): Rz(+pi/2) added first
#   BS.Rx(theta) is a primitive Perceval gate (not a true MZI)
#   verified accuracy: diff ~ 1e-16 (machine precision)
# ============================================================

def add_rz(circuit, phi):
    """Add Rz(phi) to circuit using two PS gates."""
    circuit.add(0, comp.PS(float(-phi / 2)))
    circuit.add(1, comp.PS(float( phi / 2)))

def add_ry_as_bsrx(circuit, theta):
    """
    Add BS.Ry(theta) using only hardware-native components.
    Decomposition: Rz(+pi/2) --> BS.Rx(theta) --> Rz(-pi/2)
    Verified accuracy: diff ~ 1e-16 (machine precision).
    """
    add_rz(circuit, +np.pi/2)
    circuit.add((0,1), comp.BS.Rx(theta=float(theta)))
    add_rz(circuit, -np.pi/2)

def build_qsp_pic_hardware(theta_arr, phi_arr, x_val, L):
    """
    Build full hardware-compatible QSP circuit for L layers.
    Structure:
      A(t0,p0) then L x [Rz(x), A(tj,pj)]
    Each A(t,p) = Ry(t) @ Rz(p) implemented as:
      add_rz(phi) then add_ry_as_bsrx(theta)
    Only uses BS.Rx (primitive gate) and PS. No BS.Ry used.
    """
    circuit = pcvl.Circuit(2, name=f"QSP_HW_L{L}")
    add_rz(circuit, phi_arr[0])
    add_ry_as_bsrx(circuit, theta_arr[0])
    for j in range(1, L + 1):
        add_rz(circuit, x_val)
        add_rz(circuit, phi_arr[j])
        add_ry_as_bsrx(circuit, theta_arr[j])
    return circuit

def get_pic_z_hardware(theta_arr, phi_arr, x_val, L):
    """Simulate circuit and return Z = p0-p1 (measurable output)."""
    circuit = build_qsp_pic_hardware(theta_arr, phi_arr, x_val, L)
    U       = np.array(circuit.compute_unitary())
    psi     = U @ np.array([1.0, 0.0])
    return abs(psi[0])**2 - abs(psi[1])**2

# ============================================================
# Verification: hardware circuit must match classical QSP
# All diffs must be ~0 (machine precision) for circuit to be correct
# ============================================================

print("\n========== Hardware BS.Rx Verification ==========")
for x_test in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    c_val  = classical_qsp(theta_opt, phi_opt, x_test, L)
    hw_val = get_pic_z_hardware(theta_opt, phi_opt, x_test, L)
    print(f"  x={x_test:+.1f}: classical={c_val:.6f}  "
          f"hardware={hw_val:.6f}  "
          f"diff={abs(c_val - hw_val):.2e}  "
          f"{'OK' if abs(c_val-hw_val)<1e-6 else 'WRONG'}")
print("==================================================")

# ============================================================
# Full sweep over x in [-pi, pi]
# Compares hardware circuit vs classical and vs true STEP
# ============================================================

theta_grid  = np.linspace(-np.pi, np.pi, 300)
f_classical = np.zeros(len(theta_grid))
f_hardware  = np.zeros(len(theta_grid))
f_surrogate = np.array([step_surrogate(x) for x in theta_grid])
f_true      = step_true(theta_grid)

print(f"\nSweeping {len(theta_grid)} points...")
for i, x in enumerate(theta_grid):
    f_classical[i] = classical_qsp(theta_opt, phi_opt, x, L)
    f_hardware[i]  = get_pic_z_hardware(theta_opt, phi_opt, x, L)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(theta_grid)} done")
print("Sweep complete.")

mse_hw_classic = np.mean((f_hardware - f_classical) ** 2)
mse_hw_true    = np.mean((f_hardware - f_true)      ** 2)

print("\n========== MSE Report ==========")
print(f"  MSE hardware vs classical : {mse_hw_classic:.2e}  must be ~0")
print(f"  MSE hardware vs true STEP : {mse_hw_true:.4f}")
print("=================================")

# ============================================================
# Plot: left panel shows STEP approximation quality,
# right panel shows residual between hardware and classical
# (should be zero if decomposition is correct)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Photonic QSP - Hardware BS.Rx Circuit  STEP L={L}  Z=p0-p1",
             fontsize=13)

ax = axes[0]
ax.plot(theta_grid, f_true,      'k-',  lw=2,   label="True STEP")
ax.plot(theta_grid, f_surrogate, 'g--', lw=2,   label="arctan surrogate")
ax.plot(theta_grid, f_classical, 'r-',  lw=1.5, label="Classical")
ax.plot(theta_grid, f_hardware,  'b.',  ms=3,
        label=f"Hardware BS.Rx  MSE={mse_hw_true:.4f}")
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
diff = f_hardware - f_classical
ax2.plot(theta_grid, diff, 'purple', lw=1.5,
         label=f"Hardware minus Classical  MSE={mse_hw_classic:.2e}")
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.fill_between(theta_grid, diff, alpha=0.2, color='purple')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks([-np.pi, 0, np.pi])
ax2.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"], fontsize=12)
ax2.set_xlabel(r"$x$", fontsize=12)
ax2.set_ylabel("Residual", fontsize=12)
ax2.set_title("Hardware vs Classical  must be ~0", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pic_qsp_hardware_BSRx_v2.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: pic_qsp_hardware_BSRx_v2.png")

# ============================================================
# Count hardware resources needed for Quandela experiment
# ============================================================

circ_hw = build_qsp_pic_hardware(theta_opt, phi_opt, 0.5, L)
n_ps = sum(1 for _, c in circ_hw._components if isinstance(c, comp.PS))
n_bs = sum(1 for _, c in circ_hw._components if isinstance(c, comp.BS))

print("\n========== Hardware Resource Count for Quandela ==========")
print(f"  QSP layers L                  : {L}")
print(f"  BS.Rx (primitive Perceval gate): {n_bs}")
print(f"  Phase shifters PS             : {n_ps}")
print(f"  Total components              : {n_ps + n_bs}")
print(f"  Waveguide modes               : 2  dual-rail qubit")
print(f"  Output measurement            : Z = p0-p1 photon counting")
print("===========================================================")

# ============================================================
# Manual circuit diagram (L_draw layers shown)
# Drawn from scratch to avoid Perceval label overlap issues.
# L=1 chosen for maximum readability of phase labels.
# Color coding:
#   Red    = BS.Rx (primitive Perceval gate)
#   Purple = PS for Rz(±pi/2) wrappers around each BS.Rx
#   Blue   = PS for Rz(phi) angle gates
#   Green  = PS for Rz(x) signal gates (swept during experiment)
#   Purple detector = single photon detectors
# ============================================================

def draw_qsp_circuit_hardware(theta_arr, phi_arr, L_draw=1, x_val=0.5):
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 2.8)
    ax.axis('off')

    y0, y1 = 2.0, 0.5

    def ps_box(xc, y, label, color='#4C72B0'):
        box = FancyBboxPatch((xc-0.035, y-0.18), 0.070, 0.36,
                             boxstyle="round,pad=0.004",
                             facecolor=color, edgecolor='black',
                             linewidth=0.8, zorder=3)
        ax.add_patch(box)
        ax.text(xc, y, label, ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold', zorder=4)

    def bsrx_box(xc, theta):
        """Draw BS.Rx(theta) box spanning both modes."""
        yc   = (y0 + y1) / 2
        half = (y0 - y1) / 2 + 0.12
        box  = FancyBboxPatch((xc-0.055, yc-half), 0.110, 2*half,
                              boxstyle="round,pad=0.004",
                              facecolor='#C44E52', edgecolor='black',
                              linewidth=0.8, zorder=3)
        ax.add_patch(box)
        ax.text(xc, yc, f"BS.Rx\nθ={theta:.4f}", ha='center', va='center',
                fontsize=8.5, color='white', fontweight='bold', zorder=4)

    ax.plot([0.01, 0.99], [y0, y0], 'k-', lw=1.2, zorder=1)
    ax.plot([0.01, 0.99], [y1, y1], 'k-', lw=1.2, zorder=1)
    ax.text(0.005, y0, 'mode 0', fontsize=8, va='center', ha='right')
    ax.text(0.005, y1, 'mode 1', fontsize=8, va='center', ha='right')

    n_slots = 4 + 5 * L_draw
    sw      = 0.88 / n_slots
    xc      = 0.06

    def draw_rz(xc, phi, color='#4C72B0', label=None):
        ps_box(xc, y0, f"PS\n{-phi/2:.4f}", color=color)
        ps_box(xc, y1, f"PS\n{+phi/2:.4f}", color=color)
        if label:
            ax.text(xc, y0+0.42, label, fontsize=7, ha='center',
                    color=color, style='italic')
        return xc + sw

    def draw_bsrx(xc, theta, label=None):
        bsrx_box(xc, theta)
        if label:
            ax.text(xc, y0+0.42, label, fontsize=7, ha='center',
                    color='#C44E52', style='italic')
        return xc + sw

    def draw_ry_block(xc, theta, label=None):
        """Draw Ry(theta) as Rz(+pi/2) + BS.Rx(theta) + Rz(-pi/2)."""
        xc = draw_rz(xc, +np.pi/2, color='#9467BD')
        xc = draw_bsrx(xc, theta, label=label)
        xc = draw_rz(xc, -np.pi/2, color='#9467BD')
        return xc

    # Initial block A(theta_0, phi_0)
    ax.text(xc + 1.5*sw, y0+0.65, 'A(θ₀,φ₀)', fontsize=8,
            ha='center', color='gray', style='italic')
    xc = draw_rz(xc, phi_arr[0], label='Rz(φ₀)')
    xc = draw_ry_block(xc, theta_arr[0])

    # Layers 1..L_draw
    for j in range(1, L_draw + 1):
        xc += sw * 0.4
        ax.text(xc + 2.5*sw, y0+0.65, f'A(θ{j},φ{j})', fontsize=8,
                ha='center', color='gray', style='italic')
        xc = draw_rz(xc, x_val,      color='#55A868', label='Rz(x)')
        xc = draw_rz(xc, phi_arr[j], label=f'Rz(φ{j})')
        xc = draw_ry_block(xc, theta_arr[j])

    # Detectors
    for y, lbl in [(y0, 'D₀\n(p₀)'), (y1, 'D₁\n(p₁)')]:
        box = FancyBboxPatch((0.945, y-0.18), 0.048, 0.36,
                             boxstyle="round,pad=0.004",
                             facecolor='#8172B2', edgecolor='black',
                             linewidth=0.8, zorder=3)
        ax.add_patch(box)
        ax.text(0.969, y, lbl, ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold', zorder=4)
    ax.text(0.969, y1-0.42, 'Z=p₀-p₁', fontsize=8,
            ha='center', color='purple', fontweight='bold')

    legend_elements = [
        mpatches.Patch(color='#C44E52', label='BS.Rx (primitive Perceval gate)'),
        mpatches.Patch(color='#9467BD', label='PS for Rz(±π/2) wrapper'),
        mpatches.Patch(color='#4C72B0', label='PS for Rz(φ) angle'),
        mpatches.Patch(color='#55A868', label='PS for Rz(x) signal'),
        mpatches.Patch(color='#8172B2', label='Detector'),
    ]
    ax.legend(handles=legend_elements, loc='lower center',
              ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.15))

    ax.set_title(
        f'Hardware QSP Circuit (Quandela)  L={L_draw} layer shown  |  '
        f'BS.Rx(θ) wrapped by Rz(±π/2) PS pairs',
        fontsize=9, pad=10)

    plt.tight_layout()
    plt.savefig("pic_qsp_circuit_hardware_v2.png", dpi=200, bbox_inches='tight')
    plt.show()
    print("Saved: pic_qsp_circuit_hardware_v2.png")

# Draw L=1 layer only for maximum readability
draw_qsp_circuit_hardware(theta_opt, phi_opt, L_draw=1, x_val=0.5)

# ============================================================
# Print full component list for all L=15 layers
# Shows every component, its layer, mode, type and phase value
# so you can understand the full circuit without plotting it
# ============================================================

print("\n")
print("=" * 75)
print("  FULL CIRCUIT COMPONENT LIST  (L=15, x=0.5)")
print("=" * 75)
print(f"  {'#':<5} {'Block':<24} {'Component':<10} {'Mode':<8} {'Phase (rad)':<14} {'Phase/pi'}")
print("-" * 75)

component_index = 0

def print_component(idx, block_name, comp_type, mode, phase):
    print(f"  {idx:<5} {block_name:<24} {comp_type:<10} {mode:<8} {phase:<14.6f} {phase/np.pi:.4f}π")

x_val = 0.5

# Initial block A(theta_0, phi_0)
print_component(component_index, "A(θ0,φ0) - Rz(φ0)",    "PS", "mode 0", -phi_opt[0]/2);  component_index += 1
print_component(component_index, "A(θ0,φ0) - Rz(φ0)",    "PS", "mode 1", +phi_opt[0]/2);  component_index += 1
print_component(component_index, "A(θ0,φ0) - Rz(+π/2)",  "PS", "mode 0", -np.pi/4);       component_index += 1
print_component(component_index, "A(θ0,φ0) - Rz(+π/2)",  "PS", "mode 1", +np.pi/4);       component_index += 1
print(f"  {component_index:<5} {'A(θ0,φ0) - BS.Rx':<24} {'BS.Rx':<10} {'0+1':<8} {theta_opt[0]:<14.6f} {theta_opt[0]/np.pi:.4f}π")
component_index += 1
print_component(component_index, "A(θ0,φ0) - Rz(-π/2)",  "PS", "mode 0", +np.pi/4);       component_index += 1
print_component(component_index, "A(θ0,φ0) - Rz(-π/2)",  "PS", "mode 1", -np.pi/4);       component_index += 1

print("-" * 75)

# Layers 1..L
for j in range(1, L + 1):
    print_component(component_index, f"Layer {j} - Rz(x)",       "PS", "mode 0", -x_val/2);        component_index += 1
    print_component(component_index, f"Layer {j} - Rz(x)",       "PS", "mode 1", +x_val/2);        component_index += 1
    print_component(component_index, f"Layer {j} - Rz(φ{j})",    "PS", "mode 0", -phi_opt[j]/2);   component_index += 1
    print_component(component_index, f"Layer {j} - Rz(φ{j})",    "PS", "mode 1", +phi_opt[j]/2);   component_index += 1
    print_component(component_index, f"Layer {j} - Rz(+π/2)",    "PS", "mode 0", -np.pi/4);        component_index += 1
    print_component(component_index, f"Layer {j} - Rz(+π/2)",    "PS", "mode 1", +np.pi/4);        component_index += 1
    print(f"  {component_index:<5} {f'Layer {j} - BS.Rx':<24} {'BS.Rx':<10} {'0+1':<8} {theta_opt[j]:<14.6f} {theta_opt[j]/np.pi:.4f}π")
    component_index += 1
    print_component(component_index, f"Layer {j} - Rz(-π/2)",    "PS", "mode 0", +np.pi/4);        component_index += 1
    print_component(component_index, f"Layer {j} - Rz(-π/2)",    "PS", "mode 1", -np.pi/4);        component_index += 1
    print("-" * 75)

print(f"\n  Total components       : {component_index}")
print(f"  Total PS               : {component_index - (L+1)}  (all phase shifters)")
print(f"  Total BS.Rx            : {L+1}  (one per layer + initial block)")
print("=" * 75)