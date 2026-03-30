# ============================================================
# QSP STEP FUNCTION - QUANDELA QPU EXPERIMENTAL VERSION
# ============================================================
#
# GOAL:
#   Run the QSP STEP function circuit on the real Quandela
#   photonic chip via remote connection to Quandela Cloud.
#   This produces EXPERIMENTAL results from real hardware.
#
# DIFFERENCE FROM SLOS LOCAL CODE:
#   SLOS code   : pcvl.Processor("SLOS", circuit)
#                 runs locally on your computer, no internet,
#                 no token, no QPU credits consumed
#   This code   : pcvl.RemoteProcessor("qpu:belenos", token)
#                 submits job to real Quandela photonic chip,
#                 requires token + QPU credits,
#                 results are real experimental photon counts
#
# WHAT CHANGED FROM SLOS CODE (summary at bottom of file):
#   1.  Added token setup:     pcvl.RemoteConfig.set_token(...)
#   2.  Replaced Processor:    pcvl.Processor("SLOS") ->
#                              pcvl.RemoteProcessor("qpu:belenos")
#   3.  Added max_shots_per_call to Sampler (required for remote)
#   4.  Changed job execution: sample_count(N_SHOTS) ->
#                              sample_count.execute_async(N_SHOTS)
#   5.  Added progress polling loop (async job, not instant)
#   6.  Added job ID saving (so you can resume if disconnected)
#   7.  Reduced x_values: 100 -> 25 (saves QPU credits)
#   8.  All labels changed to "experimental" / "QPU"
#   9.  Saved results use "experimental" in filename
#   10. Added credit estimation before running
#   11. Fix 1: Auto-retry up to 3 times if result is None,
#              then resubmit job if all retries fail
#   12. Fix 2: Use NaN instead of 0.0 for truly missing points
#              so they are invisible in the plot (not fake zeros)
#   13. Run-specific filenames: all saved plots and .npy files
#       include x_values count and N_shots in their name so
#       different runs never overwrite each other
#
# NAMING CONVENTIONS:
#   z_experimental       : Z = p0-p1 from real QPU hardware
#   f_perceval_analytic  : Perceval compute_unitary(), exact
#   f_slos               : local SLOS reference (if available)
#
# HOW TO RUN:
#   1. Go to cloud.quandela.com and get your API token
#   2. Replace 'YOUR_API_TOKEN_HERE' with your actual token
#   3. Replace 'qpu:belenos' with the QPU name available to you
#   4. Run: python qsp_step_experiment_qpu.py
#
# HOW TO RESUME A JOB IF DISCONNECTED:
#   If your connection drops mid-run, use the saved job IDs:
#   remote_proc = pcvl.RemoteProcessor("qpu:belenos")
#   job = remote_proc.resume_job("JOB_ID_FROM_job_ids_qpu.txt")
#   results = job.get_results()
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components as comp
from perceval.algorithm import Sampler
import time
import os

# ============================================================
# Step 1: Token setup
# ============================================================

MY_TOKEN   = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjU5MiwiZXhwIjoxODA1OTM3MjE2Ljk5ODA0NTd9.4CcQuD0_Oo9Kh7ptB-7pNFCNoM1cCy1smMicszHN3De9X88Dd05J6l89GkveCIiWwqG7H3QHI_Sq7Vk91F53Vw"   # <-- replace with your token
QPU_NAME   = "qpu:belenos"           # <-- replace with your QPU name

# ============================================================
# Step 2: Load optimized QSP angles
# ============================================================

# Using optimization angles (Step Pulse 3):
# theta_opt = np.load("theta_step_opt.npy")
# phi_opt   = np.load("phi_step_opt.npy")

# Using NLFT angles:
theta_opt = np.load("theta_step_nlft.npy")
phi_opt   = np.load("phi_step_nlft.npy")

L = len(theta_opt) - 1
print(f"Loaded QSP angles: L={L}")

N_approx = 100

def step_surrogate(x):
    """Smooth arctan approximation of STEP function."""
    return (2.0 / np.pi) * np.arctan(N_approx * x)

def step_true(x):
    """Ideal STEP function: -1 for x<0, +1 for x>=0."""
    return np.where(x >= 0, 1.0, -1.0)

def build_qsp_pic(theta_arr, phi_arr, x_val, L):
    """
    Build the Perceval QSP circuit for a specific x value.
    Structure per layer:
      Rz(phi_j) : PS(-phi_j/2) mode0 + PS(+phi_j/2) mode1  -- fixed
      Ry(theta_j): BS.Ry(theta_j)                           -- fixed
      Rz(x)     : PS(-x/2) mode0 + PS(+x/2) mode1          -- varies with x
    """
    circuit = pcvl.Circuit(2, name=f"QSP_STEP_L{L}")

    # Initial block A(theta_0, phi_0) = Ry(theta_0) * Rz(phi_0)
    circuit.add(0,      comp.PS(float(-phi_arr[0] / 2)))
    circuit.add(1,      comp.PS(float( phi_arr[0] / 2)))
    circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[0])))

    for j in range(1, L + 1):
        # Signal unitary Rz(x) -- only part that changes with x
        circuit.add(0, comp.PS(float(-x_val / 2)))
        circuit.add(1, comp.PS(float( x_val / 2)))
        # Fixed block A(theta_j, phi_j) = Ry(theta_j) * Rz(phi_j)
        circuit.add(0,      comp.PS(float(-phi_arr[j] / 2)))
        circuit.add(1,      comp.PS(float( phi_arr[j] / 2)))
        circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[j])))

    return circuit

# ============================================================
# Step 3: Experiment settings
#
# Change x_values and N_SHOTS here to run different experiments.
# All output filenames automatically include these values so
# each run saves to its own uniquely named files -- no overwriting.
# ============================================================

#N_SHOTS  = 100000   # full production run
#N_SHOTS  = 1000     # quick test
N_SHOTS  = 5000      # good balance of quality and credit cost

#x_values = np.linspace(-np.pi, np.pi, 5)    # quick test
#x_values  = np.linspace(-np.pi, np.pi, 25)   # real experiment
#x_values  = np.linspace(-np.pi, np.pi, 50)   # real experiment
x_values  = np.linspace(-np.pi, np.pi, 100)   # real experiment

# ============================================================
# Run-specific filename tag
#
# All saved files (plot + .npy) include the number of x points
# and N_shots in their name. This means running the code with
# different settings never overwrites a previous run's results.
#
# Example filenames produced:
#   x=25,  N=5000  --> qsp_step_experimental_qpu_x25_shots5000.png
#   x=100, N=50000 --> qsp_step_experimental_qpu_x100_shots50000.png
# ============================================================
RUN_TAG = f"x{len(x_values)}_shots{N_SHOTS}"
print(f"Run tag: {RUN_TAG}  -- all output files will include this tag")

# ============================================================
# Step 4: Connect to Quandela Cloud QPU
# ============================================================

print("\n" + "=" * 60)
print("  Connecting to Quandela Cloud QPU")
print("=" * 60)

pcvl.RemoteConfig.set_token(MY_TOKEN)
pcvl.RemoteConfig().save()
print(f"  Token saved.")

remote_proc_test = pcvl.RemoteProcessor(QPU_NAME)
specs = remote_proc_test.specs
print(f"  Connected to   : {QPU_NAME}")
print(f"  Max modes      : {specs['constraints']['max_mode_count']}")
print(f"  Max photons    : {specs['constraints']['max_photon_count']}")
print(f"  Our circuit    : 2 modes, 1 photon  -> OK")

print(f"\n  Estimating QPU shots needed...")
circuit_test = build_qsp_pic(theta_opt, phi_opt, 0.0, L)
remote_proc_test.set_circuit(circuit_test)
remote_proc_test.with_input(pcvl.BasicState([1, 0]))
remote_proc_test.min_detected_photons_filter(1)
required_shots = remote_proc_test.estimate_required_shots(nsamples=N_SHOTS)
total_shots    = required_shots * len(x_values)
print(f"  Shots per x point  : {required_shots:,}")
print(f"  x points           : {len(x_values)}")
print(f"  Total shots needed : {total_shots:,}")
print(f"\n  >>> Check your QPU credit balance before proceeding <<<")
print("=" * 60)

# ============================================================
# Step 5: Run experimental sweep on QPU
# ============================================================

# ============================================================
# Fix 1 setting: maximum number of retries per x point
#
# FIX 1 -- AUTO RETRY:
#   On real QPU hardware, a job sometimes returns None even
#   when it shows status SUCCESS. This happens due to transient
#   network glitches, QPU buffer issues, or cloud server timeouts.
#   Instead of immediately giving up and recording a fake zero,
#   Fix 1 retries get_results() up to MAX_RETRIES times with a
#   10-second wait between each attempt. If all retries fail,
#   it resubmits the entire job to the QPU from scratch.
#   This prevents missing data points caused by transient failures.
# ============================================================
MAX_RETRIES = 3   # Fix 1: number of retry attempts before resubmitting

# ============================================================
# Fix 2: initialize arrays with NaN instead of zeros
#
# FIX 2 -- NaN INSTEAD OF ZERO:
#   If Fix 1 exhausts all retries and still gets no result,
#   the data point is genuinely missing. Instead of storing 0.0
#   (which creates a fake outlier dot on the plot at Z=0),
#   Fix 2 stores NaN (Not a Number). Matplotlib automatically
#   skips NaN values when plotting, so the missing point simply
#   disappears from the figure rather than appearing as a wrong
#   value. This keeps the plot clean and scientifically honest.
#   np.nanmean() is used later so NaN points are excluded from
#   the MSE calculation as well.
# ============================================================
z_experimental  = np.full(len(x_values), np.nan)   # Fix 2: NaN not 0.0
p0_experimental = np.full(len(x_values), np.nan)   # Fix 2: NaN not 0.0
p1_experimental = np.full(len(x_values), np.nan)   # Fix 2: NaN not 0.0

# Job ID file also tagged with run parameters
job_id_file = f"job_ids_qpu_{RUN_TAG}.txt"
with open(job_id_file, "w") as f:
    f.write(f"QSP STEP QPU Experiment -- {QPU_NAME}\n")
    f.write(f"Run tag        : {RUN_TAG}\n")
    f.write(f"x_values       : {len(x_values)} points\n")
    f.write(f"N_SHOTS        : {N_SHOTS}\n")
    f.write(f"Fix 1 retries  : {MAX_RETRIES}\n")
    f.write(f"Fix 2 NaN      : enabled\n\n")

print(f"\nStarting QPU experimental sweep:")
print(f"  {len(x_values)} x values, {N_SHOTS} shots each")
print(f"  Fix 1 active: up to {MAX_RETRIES} retries + resubmit per x point")
print(f"  Fix 2 active: missing points stored as NaN (not 0.0)")
print(f"  Run tag: {RUN_TAG}")
print(f"  Job IDs saved to: {job_id_file}")
print("=" * 60)

for i, x_val in enumerate(x_values):

    print(f"\n  [{i+1:2d}/{len(x_values)}] x = {x_val:+.4f} rad")

    circuit = build_qsp_pic(theta_opt, phi_opt, x_val, L)

    remote_proc = pcvl.RemoteProcessor(QPU_NAME)
    remote_proc.set_circuit(circuit)
    remote_proc.with_input(pcvl.BasicState([1, 0]))
    remote_proc.min_detected_photons_filter(1)

    sampler = Sampler(remote_proc, max_shots_per_call=N_SHOTS)

    job = sampler.sample_count.execute_async(N_SHOTS)

    with open(job_id_file, "a") as f:
        f.write(f"x[{i:02d}] = {x_val:+.4f}  job_id = {job.id}\n")
    print(f"    Job submitted. ID: {job.id}")

    print(f"    Waiting for QPU result", end="", flush=True)
    while not job.is_complete:
        time.sleep(5)
        print(".", end="", flush=True)
    print(f" done.")
    print(f"    Job status: {job.status()}")

    # --------------------------------------------------------
    # Fix 1 -- AUTO RETRY:
    #   Try get_results() up to MAX_RETRIES times.
    #   Wait 10 seconds between each attempt.
    #   If all retries fail, resubmit the entire job once more.
    #   This handles transient QPU/network failures automatically
    #   without losing the data point.
    # --------------------------------------------------------
    results = None
    for attempt in range(MAX_RETRIES):
        results = job.get_results()
        if results is not None and results.get('results') is not None:
            break   # Fix 1: got valid results, stop retrying
        print(f"    Fix 1: result was None, retry "
              f"{attempt+1}/{MAX_RETRIES} in 10s...")
        time.sleep(10)

    # Fix 1: if all retries failed, resubmit the job entirely
    if results is None or results.get('results') is None:
        print(f"    Fix 1: all {MAX_RETRIES} retries failed "
              f"-- resubmitting job to QPU")
        job2 = sampler.sample_count.execute_async(N_SHOTS)
        with open(job_id_file, "a") as f:
            f.write(f"  RESUBMIT x[{i:02d}] job_id = {job2.id}\n")
        print(f"    Fix 1: resubmitted. New job ID: {job2.id}")
        print(f"    Fix 1: waiting for resubmitted job",
              end="", flush=True)
        while not job2.is_complete:
            time.sleep(5)
            print(".", end="", flush=True)
        print(" done.")
        results = job2.get_results()

    # --------------------------------------------------------
    # Fix 2 -- NaN FOR TRULY MISSING DATA:
    #   If Fix 1 exhausted all retries AND the resubmitted job
    #   also returned None, the data point is genuinely missing.
    #   Store NaN instead of 0.0 so matplotlib skips it in the
    #   plot (no fake outlier dot at Z=0 in the figure).
    #   The arrays were initialized with NaN above so no action
    #   is needed here -- just skip to the next x point.
    # --------------------------------------------------------
    if results is None or results.get('results') is None:
        print(f"    Fix 2: storing NaN for x={x_val:.4f} "
              f"-- missing point will be invisible in plot")
        continue   # Fix 2: arrays already contain NaN at index i

    counts = dict(results['results'])

    count_mode0 = counts.get(pcvl.BasicState([1, 0]), 0)
    count_mode1 = counts.get(pcvl.BasicState([0, 1]), 0)
    total       = count_mode0 + count_mode1

    if total > 0:
        p0 = count_mode0 / total
        p1 = count_mode1 / total
        z  = p0 - p1
    else:
        p0, p1, z = 0.0, 0.0, 0.0
        print(f"    WARNING: no counts detected at x={x_val:.4f}")

    z_experimental[i]  = z
    p0_experimental[i] = p0
    p1_experimental[i] = p1

    print(f"    mode0={count_mode0:6d}  mode1={count_mode1:6d}  "
          f"p0={p0:.4f}  p1={p1:.4f}  Z={z:+.4f}")

print("\n" + "=" * 60)
print("Experimental sweep complete.")

n_valid   = int(np.sum(~np.isnan(z_experimental)))
n_missing = int(np.sum( np.isnan(z_experimental)))
print(f"  Valid data points   : {n_valid} / {len(x_values)}")
print(f"  Missing data points : {n_missing} / {len(x_values)}  "
      f"(NaN -- invisible in plot)")

# ============================================================
# Step 6: Save experimental results
#
# All filenames include RUN_TAG (x count + N_shots) so each
# run saves to uniquely named files and never overwrites a
# previous run's results.
# ============================================================

np.save(f"x_values_experimental_{RUN_TAG}.npy", x_values)
np.save(f"z_experimental_{RUN_TAG}.npy",        z_experimental)
np.save(f"p0_experimental_{RUN_TAG}.npy",       p0_experimental)
np.save(f"p1_experimental_{RUN_TAG}.npy",       p1_experimental)
print(f"\nExperimental results saved:")
print(f"  x_values_experimental_{RUN_TAG}.npy")
print(f"  z_experimental_{RUN_TAG}.npy")
print(f"  p0_experimental_{RUN_TAG}.npy")
print(f"  p1_experimental_{RUN_TAG}.npy")

# ============================================================
# Step 7: Compute reference curves for comparison
# ============================================================

x_fine      = np.linspace(-np.pi, np.pi, 300)
f_surrogate = np.array([step_surrogate(x) for x in x_fine])
f_true      = step_true(x_fine)

print("\nComputing Perceval analytic reference...")
f_perceval_analytic = np.zeros(len(x_values))
for i, x_val in enumerate(x_values):
    circuit = build_qsp_pic(theta_opt, phi_opt, x_val, L)
    U = np.array(circuit.compute_unitary())
    psi = U @ np.array([1.0, 0.0])
    f_perceval_analytic[i] = abs(psi[0])**2 - abs(psi[1])**2

slos_available = False
if os.path.exists("z_slos.npy") and os.path.exists("x_values_slos.npy"):
    x_slos = np.load("x_values_slos.npy")
    z_slos = np.load("z_slos.npy")
    slos_available = True
    print("SLOS local results loaded for comparison.")

# ============================================================
# Step 8: MSE report
#
# np.nanmean is used instead of np.mean so that NaN points
# (Fix 2) are automatically excluded from the MSE calculation.
# This gives a fair MSE over valid data points only.
# ============================================================

mse_exp_vs_analytic  = np.nanmean(
    (z_experimental - f_perceval_analytic)**2)
mse_exp_vs_surrogate = np.nanmean(
    (z_experimental - np.array([step_surrogate(x) for x in x_values]))**2)
mse_exp_vs_true      = np.nanmean(
    (z_experimental - step_true(x_values))**2)

print(f"\n========== Experimental MSE Report  [{RUN_TAG}] ==========")
print(f"  Valid points used for MSE  : {n_valid} / {len(x_values)}")
print(f"  MSE experimental vs Perceval analytic  : {mse_exp_vs_analytic:.4f}")
print(f"    (hardware noise + imperfections)")
print(f"  MSE experimental vs surrogate (arctan) : {mse_exp_vs_surrogate:.4f}")
print(f"  MSE experimental vs true STEP          : {mse_exp_vs_true:.4f}")
print(f"=====================================================")

# ============================================================
# Step 9: Plot experimental results
#
# The plot filename includes RUN_TAG so each run produces a
# uniquely named figure file. For example:
#   x=25,  N=5000  --> qsp_step_experimental_qpu_x25_shots5000.png
#   x=100, N=50000 --> qsp_step_experimental_qpu_x100_shots50000.png
#
# Fix 2 effect on plot:
#   matplotlib automatically skips NaN values when plotting.
#   Missing data points simply do not appear as dots --
#   no fake Z=0 outlier appears in the figure.
# ============================================================

ncols = 3 if slos_available else 2
fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 5))
fig.suptitle(
    f"QSP STEP Function -- Experimental Results  "
    f"QPU: {QPU_NAME}  L={L}  "
    f"x points={len(x_values)}  N_shots={N_SHOTS}",
    fontsize=12, fontweight='bold'
)

xt = [-np.pi, 0, np.pi]
xl = [r"$-\pi$", r"$0$", r"$\pi$"]

# Left panel: experimental results
ax = axes[0]
ax.plot(x_fine,   f_true,              'k-',  lw=2.5,
        label="True STEP",                          zorder=3)
ax.plot(x_fine,   f_surrogate,         'g--', lw=1.5,
        label="arctan surrogate",                   zorder=2)
ax.plot(x_values, f_perceval_analytic, 'r-',  lw=1.5,
        label="Perceval analytic  Z=p0-p1",         zorder=4)
ax.plot(x_values, z_experimental,      'b.',  ms=10,
        label=f"Experimental  Z=p0-p1\n"
              f"  MSE vs surrogate={mse_exp_vs_surrogate:.4f}\n"
              f"  MSE vs true STEP={mse_exp_vs_true:.4f}",
        zorder=5)
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.3, 1.3])
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=11)
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel("Z = p0 - p1", fontsize=12)
ax.set_title(f"Experimental QPU results  L={L}", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Middle panel: residual experimental vs Perceval analytic
# Fix 2: NaN points produce no dot and no line segment here
ax2 = axes[1]
diff_exp = z_experimental - f_perceval_analytic
ax2.plot(x_values, diff_exp, color='darkred', lw=1.5, marker='.', ms=8,
         label=f"Experimental minus Perceval analytic\n"
               f"  MSE vs Perceval analytic={mse_exp_vs_analytic:.4f}\n"
               f"  (hardware noise + imperfections)")
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.fill_between(x_values, diff_exp,
                 where=~np.isnan(diff_exp),   # Fix 2: skip NaN in fill
                 alpha=0.2, color='darkred')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks(xt); ax2.set_xticklabels(xl, fontsize=11)
ax2.set_xlabel(r"$x$", fontsize=12)
ax2.set_ylabel("Residual", fontsize=12)
ax2.set_title("Experimental vs Perceval analytic residual\n"
              "(shows real hardware noise)", fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Right panel: 3-way comparison (only if SLOS results exist)
if slos_available:
    ax3 = axes[2]
    ax3.plot(x_fine,   f_true,              'k-',  lw=2.5,
             label="True STEP",                    zorder=3)
    ax3.plot(x_values, f_perceval_analytic, 'g--', lw=2,
             label="Perceval analytic",             zorder=4)
    ax3.plot(x_slos,   z_slos,              'b.',  ms=6,
             label="SLOS local simulation",         zorder=5)
    ax3.plot(x_values, z_experimental,      'r.',  ms=10,
             label=f"Experimental (QPU: {QPU_NAME})", zorder=6)
    ax3.set_xlim([-np.pi, np.pi])
    ax3.set_ylim([-1.3, 1.3])
    ax3.set_xticks(xt); ax3.set_xticklabels(xl, fontsize=11)
    ax3.set_xlabel(r"$x$", fontsize=12)
    ax3.set_title("Experimental vs SLOS vs Analytic\n"
                  "(3-way comparison)", fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

plt.tight_layout()

# Save plot with run-specific filename
plot_filename = f"qsp_step_experimental_qpu_{RUN_TAG}.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved: {plot_filename}")

# ============================================================
# SUMMARY OF CHANGES FROM SLOS CODE
# ============================================================
print("\n" + "=" * 60)
print("  SUMMARY OF CHANGES FROM SLOS LOCAL CODE")
print("=" * 60)
changes = [
    ("1",  "Token setup",
     "Added pcvl.RemoteConfig.set_token() + save()"),
    ("2",  "Processor",
     'pcvl.Processor("SLOS") -> pcvl.RemoteProcessor("qpu:belenos")'),
    ("3",  "Sampler",
     "Added max_shots_per_call=N_SHOTS (required for remote)"),
    ("4",  "Job execution",
     "sample_count(N) -> sample_count.execute_async(N)"),
    ("5",  "Wait loop",
     "Added polling loop while not job.is_complete"),
    ("6",  "Job ID saving",
     "Job IDs saved to job_ids_qpu_{RUN_TAG}.txt"),
    ("7",  "x_values",
     "100 points -> 25 points (saves QPU credits)"),
    ("8",  "Labels",
     "All labels now say 'experimental' / 'QPU'"),
    ("9",  "Saved files",
     "All .npy files include RUN_TAG in filename"),
    ("10", "Credit estimation",
     "Added estimate_required_shots() before running"),
    ("11", "3-way plot",
     "Right panel compares experimental vs SLOS vs analytic"),
    ("12", "Fix 1: Auto-retry",
     f"Retries get_results() up to {MAX_RETRIES}x, resubmits if still None"),
    ("13", "Fix 2: NaN not zero",
     "Missing points stored as NaN -- invisible in plot"),
    ("14", "Run-specific filenames",
     f"Plot + .npy files include '{RUN_TAG}' -- no overwriting"),
]
for num, name, desc in changes:
    print(f"  {num:>2}. {name:<25} {desc}")
print("=" * 60)