import numpy as np
from scipy.stats import t

class Paper2UltimateFrozenFinal:
def __init__(self, num_subjects=10000, steps=500, seeds=list(range(1, 21))):
self.num_subjects = num_subjects
self.steps = steps
self.seeds = seeds # 20 independent Monte Carlo replicates
self.n_seeds = len(seeds)
self.t_crit = t.ppf(0.975, self.n_seeds - 1) # t-distribution critical value (df=19 -> ~2.093)

def run_single_experiment(self, mode, toxicity_budget=0.5, model_mismatch_beta_delta=0.0, 
sensor_noise_scale=1.0, controller_bias=0.0, seed=42):
# Common Random Number (CRN) generator initialized strictly by seed
cond_rng = np.random.default_rng(seed)

# True heterogeneous parameters (re-generated per seed for MC variation)
r_aging_true = np.clip(cond_rng.normal(0.02, 0.005, self.num_subjects), 0.005, None)
beta_rep_true = np.clip(cond_rng.normal(0.05, 0.01, self.num_subjects), 0.01, None)
c_tox_true = np.clip(cond_rng.normal(0.01, 0.002, self.num_subjects), 0.001, None)

A = np.zeros(self.num_subjects)
D_rev = np.zeros(self.num_subjects)
D_irr = np.zeros(self.num_subjects)
B = np.zeros(self.num_subjects) + 30.0

toxicity_activation_count = 0
magnitude_activation_count = 0
any_activation_count = 0
subject_ever_unsafe = np.zeros(self.num_subjects, dtype=bool)

# Predefined normalized computational safety thresholds
D_rev_max = 50.0
D_irr_max = 20.0
B_max = 100.0

# Computational safe-operation survival tracking (T_f is right-censored at steps)
failure_times = np.full(self.num_subjects, self.steps + 1, dtype=int)
has_failed = np.zeros(self.num_subjects, dtype=bool)

# Controller-assumed parameters (Restricted strictly to repair-efficacy mismatch)
beta_rep_controller = beta_rep_true * (1.0 + model_mismatch_beta_delta)
c_tox_controller = c_tox_true 

for t in range(self.steps):
d_stress = cond_rng.exponential(0.01, self.num_subjects)
epsilon_B = cond_rng.normal(0, 0.05, self.num_subjects)

# --- Measurement-Based State Estimation with Non-negative constraints ---
sensor_noise_A = cond_rng.normal(0, 0.02 * sensor_noise_scale, self.num_subjects)
sensor_noise_D = cond_rng.normal(0, 0.05 * sensor_noise_scale, self.num_subjects)

A_hat = np.maximum(0.0, A + sensor_noise_A)
D_rev_hat = np.maximum(0.0, D_rev + sensor_noise_D)

# Control input architecture
if mode == 'baseline':
u_t = np.zeros(self.num_subjects)
elif mode == 'fixed':
u_t = np.ones(self.num_subjects) * 0.3
elif mode in ['qid_os', 'qid_uas_os']:
base_demand = 1.2 * D_rev_hat + 0.8 * A_hat
safe_gain_multiplier = max(0.0, 1.0 + controller_bias) # Safe clamp against negative gain
u_target = (base_demand / np.maximum(beta_rep_controller, 1e-4) * 0.05) * safe_gain_multiplier
u_target = np.maximum(0.0, u_target) # Non-negative control demand

if mode == 'qid_os':
u_t = u_target
else:
# UAS-OS: Dual safety supervision with separated activation metrics
max_allowed_tox = toxicity_budget / np.maximum(c_tox_controller, 1e-4)
max_allowed_mag = 1.5

is_tox_clipped = u_target > max_allowed_tox
is_mag_clipped = u_target > max_allowed_mag
any_clipped = is_tox_clipped | is_mag_clipped

toxicity_activation_count += np.sum(is_tox_clipped)
magnitude_activation_count += np.sum(is_mag_clipped)
any_activation_count += np.sum(any_clipped)

max_allowed_u = np.minimum(max_allowed_tox, max_allowed_mag)
u_t = np.minimum(u_target, max_allowed_u)

# --- Nonlinear surrogate plant dynamics (Synchronous pre-update evaluation) ---
D_crit = 2.0
transition = np.maximum(0, D_rev - D_crit)
gamma_feedback = 0.15

A_next = A + r_aging_true + d_stress - (0.5 * beta_rep_true * u_t) + gamma_feedback * D_irr
D_rev_next = D_rev + (0.1 * A) + d_stress - (beta_rep_true * u_t)
D_irr_next = D_irr + 0.2 * transition
B_next = B + (0.05 * A) + (0.1 * D_irr) - (0.03 * beta_rep_true * u_t) + epsilon_B

# --- 1. Safety evaluation evaluated strictly on UN-CLIPPED raw states ---
is_step_unsafe = (D_rev_next > D_rev_max) | (D_irr_next > D_irr_max) | (B_next > B_max)
subject_ever_unsafe |= is_step_unsafe

# Track first failure time T_f (right-censored)
newly_failed = is_step_unsafe & (~has_failed)
failure_times[newly_failed] = t + 1
has_failed |= is_step_unsafe

# --- 2. Numerical stabilization clipping applied AFTER safety checks ---
A = np.clip(A_next, -50.0, 50.0)
D_rev = np.clip(D_rev_next, 0.0, 200.0)
D_irr = np.clip(D_irr_next, 0.0, 200.0)
B = np.clip(B_next, 0.0, 500.0)

total_decisions = self.num_subjects * self.steps
activation_rate_any = any_activation_count / total_decisions
activation_rate_tox = toxicity_activation_count / total_decisions
activation_rate_mag = magnitude_activation_count / total_decisions

survival_curve = np.array([np.mean(failure_times > t) for t in range(self.steps + 1)])

return {
"mean_B": np.mean(B), "p95_B": np.percentile(B, 95),
"mean_D_irr": np.mean(D_irr), "p95_D_irr": np.percentile(D_irr, 95),
"activation_rate_any": activation_rate_any,
"activation_rate_tox": activation_rate_tox,
"activation_rate_mag": activation_rate_mag,
"P_ever_unsafe": np.mean(subject_ever_unsafe),
"survival_curve": survival_curve
}

def evaluate_multi_seed(self, **kwargs):
metrics = {
k: [] for k in ["mean_B", "p95_B", "mean_D_irr", "p95_D_irr", 
"activation_rate_any", "activation_rate_tox", 
"activation_rate_mag", "P_ever_unsafe"]
}
survival_curves = []

for seed in self.seeds:
res = self.run_single_experiment(seed=seed, **kwargs)
for k in metrics:
metrics[k].append(res[k])
survival_curves.append(res["survival_curve"])

summary = {}
for k in metrics:
arr = np.array(metrics[k])
mean_val = np.mean(arr)
ci_95 = self.t_crit * np.std(arr, ddof=1) / np.sqrt(self.n_seeds)
summary[k] = (mean_val, ci_95)

surv_arr = np.array(survival_curves)
summary["survival_mean"] = np.mean(surv_arr, axis=0)
summary["survival_ci"] = self.t_crit * np.std(surv_arr, axis=0, ddof=1) / np.sqrt(self.n_seeds)

return summary

def evaluate_paired_metric(self, metric, mode_a='qid_uas_os', mode_b='baseline', **kwargs):
"""Generalized seed-by-seed paired difference analysis using Common Random Numbers"""
a_vals = []
b_vals = []
for seed in self.seeds:
res_a = self.run_single_experiment(mode=mode_a, seed=seed, **kwargs)
res_b = self.run_single_experiment(mode=mode_b, seed=seed, **kwargs)
a_vals.append(res_a[metric])
b_vals.append(res_b[metric])

diffs = np.asarray(a_vals) - np.asarray(b_vals)
mean_diff = np.mean(diffs)
ci_diff = self.t_crit * np.std(diffs, ddof=1) / np.sqrt(self.n_seeds)
return mean_diff, ci_diff

if __name__ == "__main__":
production_seeds = list(range(1, 21))
sim = Paper2UltimateFrozenFinal(num_subjects=10000, steps=500, seeds=production_seeds)

print("=" * 90)
print(f" Paper 2 Ultimate Frozen Simulation (N={sim.num_subjects} subjects x {sim.n_seeds} seeds, t-dist CI)")
print("=" * 90)

# Table 1: Architecture Benchmark
print("\n--- Table 1: Architecture Benchmark (Mean ± t-distribution 95% CI, df=19) ---")
for mode in ['baseline', 'fixed', 'qid_os', 'qid_uas_os']:
res = sim.evaluate_multi_seed(mode=mode, toxicity_budget=0.5)
print(f"[{mode.upper():<12}] B(Mean): {res['mean_B'][0]:5.1f} (±{res['mean_B'][1]:4.2f}) | "
f"P_ever_unsafe: {res['P_ever_unsafe'][0]:.3f} (±{res['P_ever_unsafe'][1]:.3f}) | "
f"Act.Rate(Any): {res['activation_rate_any'][0]*100:4.1f}% "
f"(Tox: {res['activation_rate_tox'][0]*100:4.1f}%, Mag: {res['activation_rate_mag'][0]*100:4.1f}%)")

# Generalized Paired Comparisons (QID-UAS-OS vs Baseline)
print("\n--- Generalized Paired Comparisons (QID-UAS-OS minus Baseline under CRN) ---")
diff_b, ci_b = sim.evaluate_paired_metric("mean_B", "qid_uas_os", "baseline", toxicity_budget=0.5)
diff_p, ci_p = sim.evaluate_paired_metric("P_ever_unsafe", "qid_uas_os", "baseline", toxicity_budget=0.5)
diff_d, ci_d = sim.evaluate_paired_metric("mean_D_irr", "qid_uas_os", "baseline", toxicity_budget=0.5)
print(f"Δ Mean B : {diff_b:6.2f} (±95% CI: {ci_b:4.2f})")
print(f"Δ P_ever_unsafe : {diff_p:6.3f} (±95% CI: {ci_p:4.3f})")
print(f"Δ Mean D_irr : {diff_d:6.2f} (±95% CI: {ci_d:4.2f})")

# Hierarchical Architecture Comparison (QID-UAS-OS vs QID-OS)
print("\n--- Hierarchical Architecture Comparison (QID-UAS-OS minus QID-OS) ---")
diff_hier_p, ci_hier_p = sim.evaluate_paired_metric("P_ever_unsafe", "qid_uas_os", "qid_os", toxicity_budget=0.5)
diff_hier_b, ci_hier_b = sim.evaluate_paired_metric("mean_B", "qid_uas_os", "qid_os", toxicity_budget=0.5)
print(f"Δ P_ever_unsafe (UAS-OS vs QID-OS): {diff_hier_p:6.3f} (±95% CI: {ci_hier_p:4.3f})")
print(f"Δ Mean B (UAS-OS vs QID-OS): {diff_hier_b:6.2f} (±95% CI: {ci_hier_b:4.2f})")

# Table 2: Sensor Noise Robustness
print("\n--- Table 2: Sensor Noise Robustness Sweep (QID-UAS-OS) ---")
for ns in [0.5, 1.0, 2.0, 4.0]:
res = sim.evaluate_multi_seed(mode='qid_uas_os', toxicity_budget=0.5, sensor_noise_scale=ns)
print(f"[Noise Scale = {ns:3.1f}] B(Mean): {res['mean_B'][0]:5.1f} | "
f"P_ever_unsafe: {res['P_ever_unsafe'][0]:.3f} (±{res['P_ever_unsafe'][1]:.3f}) | "
f"Act.Rate(Any): {res['activation_rate_any'][0]*100:4.1f}%")

# Table 3: Toxicity Budget Sensitivity
print("\n--- Table 3: Toxicity Budget Sensitivity Sweep (QID-UAS-OS) ---")
for tb in [0.1, 0.3, 0.5, 1.0]:
res = sim.evaluate_multi_seed(mode='qid_uas_os', toxicity_budget=tb)
print(f"[Toxicity Budget = {tb:3.1f}] B(Mean): {res['mean_B'][0]:5.1f} | "
f"P_ever_unsafe: {res['P_ever_unsafe'][0]:.3f} (±{res['P_ever_unsafe'][1]:.3f}) | "
f"Act.Rate(Any): {res['activation_rate_any'][0]*100:4.1f}% "
f"(Tox: {res['activation_rate_tox'][0]*100:4.1f}%)")

# Table 4: Repair-Efficacy Model Mismatch
print("\n--- Table 4: Repair-Efficacy Model Mismatch Sweep (QID-UAS-OS) ---")
for d in [-0.2, 0.0, 0.2]:
res = sim.evaluate_multi_seed(mode='qid_uas_os', toxicity_budget=0.5, model_mismatch_beta_delta=d)
print(f"[Repair-Efficacy Mismatch d_beta = {d:+4.1f}] B(Mean): {res['mean_B'][0]:5.1f} | "
f"P_ever_unsafe: {res['P_ever_unsafe'][0]:.3f} (±{res['P_ever_unsafe'][1]:.3f}) | "
f"Act.Rate(Any): {res['activation_rate_any'][0]*100:4.1f}%")

# Computational Safe-Operation Survival Analysis Summary (Right-censored at T=500)
res_surv_summary = sim.evaluate_multi_seed(mode='qid_uas_os', toxicity_budget=0.5)
print("\n--- Computational Safe-Operation Survival S(t) Sample (Right-censored) ---")
print(f"S(t=0) = {res_surv_summary['survival_mean'][0]:.3f}")
print(f"S(t=100) = {res_surv_summary['survival_mean'][100]:.3f} (±{res_surv_summary['survival_ci'][100]:.3f})")
print(f"S(t=250) = {res_surv_summary['survival_mean'][250]:.3f} (±{res_surv_summary['survival_ci'][250]:.3f})")
print(f"S(t=500) = {res_surv_summary['survival_mean'][500]:.3f} (±{res_surv_summary['survival_ci'][500]:.3f})")
print("=" * 90)
