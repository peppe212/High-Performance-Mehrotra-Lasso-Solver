"""
================================================================================
Experimental Analysis Script (Chapter 7)
Reproducibility Code for Lasso IPM Solver
================================================================================

Author:         Giuseppe Muschetta
ID:             564026
Program:        M.Sc. in Data Science & Business Informatics
University:     University of Pisa
Course:         Optimization for Data Science
Instructor:     Prof. Antonio Frangioni
Academic Year:  2024/2025
Final Grade:    30/30


DESCRIPTION:
------------
This script runs the experiments described in the project report.
It compares the custom Primal-Dual IPM Solver against CVXPY (using OSQP/SCS)
and validates the solutions against Scikit-Learn's implementation [Appendix B].

The script performs three main steps:
1. Sensitivity Analysis: Testing convergence with varying budget 't' [Sec. 7.3].
2. Benchmark: Comparing time and accuracy against CVXPY [Sec. 7.4].
3. Plots: Generating convergence graphs and results [Sec. 7.5].

Datasets:
- Diabetes (Small-scale validation)
- California Housing (Scalability)
- IMDb (High-dimensional/Sparse test)
- Synthetic (Controlled test, n=2000 features)
"""

# --------------------------------------------------------------------------
# IMPORTS & ENVIRONMENT SETUP
# --------------------------------------------------------------------------
import sys
import time
import warnings
import numpy as np
import pandas as pd
import scipy
import sklearn
import cvxpy as cp
import osqp
import scs
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import custom solver implementation
try:
    from lasso_ipm_solver import lasso_ipm_solver_debug
except ImportError:
    raise ImportError("Error: 'lasso_ipm_solver.py' not found.")

# Suppress warnings for cleaner log output during batch execution
warnings.filterwarnings('ignore')

# Graphic style configuration for report-ready plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# --------------------------------------------------------------------------
# AUX FUNCTIONS
# --------------------------------------------------------------------------

def print_environment_info():
    """
    Prints the execution context and library versions for reproducibility.
    """
    print("=" * 65)
    print("        PROJECT AND EXECUTION ENVIRONMENT DETAILS")
    print("=" * 65)

    print("\n--- Project Details ---")
    print(f"{'Author:':<25} Giuseppe Muschetta")
    print(f"{'University:':<25} University of Pisa")
    print(f"{'M.Sc. Program:':<25} Data Science & Business Informatics")
    print(f"{'Course:':<25} Optimization for Data Science")
    print(f"{'Instructor:':<25} Prof. Antonio Frangioni")
    print(f"{'Academic Year:':<25} 2024/2025")
    print(f"{'Date:':<25} August 2025")

    print("\n--- Execution Environment ---")
    print(f"{'Python Version:':<25} {sys.version.split(' ')[0]}")
    print(f"{'NumPy Version:':<25} {np.__version__}")
    print(f"{'SciPy Version:':<25} {scipy.__version__}")
    print(f"{'Pandas Version:':<25} {pd.__version__}")
    print(f"{'Scikit-learn Version:':<25} {sklearn.__version__}")
    print(f"{'CVXPY Version:':<25} {cp.__version__}")
    print(f"{'OSQP Version:':<25} {osqp.__version__}")
    print(f"{'SCS Version:':<25} {scs.__version__}")
    print("=" * 65)


def print_compact_report(res, feature_names):
    """
    Outputs a structured comparison table matching the format used in
    Tables 8-15 of the project report.
    """
    print(f"\n{'=' * 80}")
    print(
        f"Dataset: {res['dataset']} | Samples: {res['dataset_info']['n_samples']} train, {res['dataset_info']['n_test']} test | Features: {res['dataset_info']['n_features']}")
    print(
        f"Budget L1: t={res['problem_setup']['t_budget']:.6f} ({res['problem_setup']['t_ratio'] * 100:.1f}% of reference)")

    if not (res['ipm']['success'] and res['cvxpy']['success']):
        ipm_status = "[OK]" if res['ipm']['success'] else "[FAIL]"
        cvxpy_status = "[OK]" if res['cvxpy']['success'] else "[FAIL]"
        print(f"Solver Status: IPM={ipm_status} | CVXPY={cvxpy_status}")
        print(f"{'=' * 80}\n")
        return

    # Compute comparative metrics
    speedup = res['cvxpy']['time_s'] / res['ipm']['time_s'] if res['ipm']['time_s'] > 0 else 0
    sol_dist = np.linalg.norm(res['ipm']['solution_vector'] - res['cvxpy']['solution_vector'])

    print(f"{'=' * 80}")

    # Results Table Construction
    data = {
        'Metric': [
            'Time (s)',
            'Iterations',
            'Speedup',
            'L1 Norm',
            'L1 Budget (t)',
            'L1 Budget Used (%)',
            'Sparsity',
            'MSE Train',
            'MSE Test',
            'MAE Test',
            'R2 Test',
            'Solution Distance L2',
            'Solver Failures',
            'Convergence'
        ],
        'IPM': [
            f"{res['ipm']['time_s']:.4f}",
            res['ipm']['iterations'],
            f"{speedup:.1f}x",
            f"{res['ipm']['l1_norm']:.6f}",
            f"{res['problem_setup']['t_budget']:.6f}",
            f"{res['ipm']['l1_utilization'] * 100:.1f}",
            f"{res['ipm']['sparsity']}/{res['dataset_info']['n_features']}",
            f"{res['ipm']['mse_train']:.6f}",
            f"{res['ipm']['mse_test']:.6f}",
            f"{res['ipm']['mae_test']:.6f}",
            f"{res['ipm']['r2_test']:.4f}",
            '-',
            res['ipm']['solver_failures'],
            res['ipm']['convergence_reason']
        ],
        f"CVXPY({res['cvxpy']['solver_used']})": [
            f"{res['cvxpy']['time_s']:.4f}",
            res['cvxpy']['iterations'],
            '-',
            f"{res['cvxpy']['l1_norm']:.6f}",
            f"{res['problem_setup']['t_budget']:.6f}",
            f"{res['cvxpy']['l1_utilization'] * 100:.1f}",
            f"{res['cvxpy']['sparsity']}/{res['dataset_info']['n_features']}",
            f"{res['cvxpy']['mse_train']:.6f}",
            f"{res['cvxpy']['mse_test']:.6f}",
            f"{res['cvxpy']['mae_test']:.6f}",
            f"{res['cvxpy']['r2_test']:.4f}",
            f"{sol_dist:.2e}",
            '-',
            'optimal'
        ]
    }

    df = pd.DataFrame(data)
    print(df.to_string(index=False))

    # Feature Selection Analysis
    print(f"\nActive Features (coeff != 0):")
    active_ipm = np.abs(res['ipm']['solution_vector']) > 1e-6
    active_cvx = np.abs(res['cvxpy']['solution_vector']) > 1e-6
    active_any = active_ipm | active_cvx

    if np.any(active_any):
        # Truncate output for high-dimensional datasets to keep logs readable
        limit_feat = 20
        indices = np.where(active_any)[0]
        for idx, i in enumerate(indices):
            if idx >= limit_feat:
                print(f"  ... and {len(indices) - limit_feat} more active features.")
                break
            fname = feature_names[i] if feature_names and i < len(feature_names) else f"feat_{i}"
            ipm_val = res['ipm']['solution_vector'][i]
            cvx_val = res['cvxpy']['solution_vector'][i]
            print(
                f"  {fname:<25} IPM: {ipm_val:>10.6f}   CVXPY: {cvx_val:>10.6f}   Diff: {abs(ipm_val - cvx_val):>9.2e}")
    else:
        print("  No active features (zero solution)")

    print(f"{'=' * 80}\n")


# --------------------------------------------------------------------------
# SINGLE EXPERIMENT RUNNER
# --------------------------------------------------------------------------

def run_experiment(dataset_name, data_loader_func,
                   t_value=None,
                   system_method="cholesky",
                   cg_max_iter=1000,
                   max_iter=200,
                   random_seed=42,
                   verbosity=False):
    """
    Executes a rigorous benchmark on a specific dataset.

    Steps:
    1. Data Loading & Preprocessing (Standard Scaling).
    2. Reference Calculation: Estimates 't' using Sklearn if not provided.
    3. IPM Execution: Solves the problem using our custom implementation.
    4. Benchmark Execution: Solves the problem using CVXPY (OSQP/SCS).
    5. Comparison: Computes accuracy (L2 distance) and efficiency (Speedup).
    """
    print("\n" + "=" * 80)
    print(f"  EXPERIMENT: {dataset_name.upper()}")
    print("=" * 80)

    # Data Preparation
    X, y, feature_names = data_loader_func()
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_seed
    )

    # Standardization: Essential for Interior Point Methods stability
    if dataset_name != 'IMDb':
        scaler_X = StandardScaler().fit(X_train)
        X_train_scaled = scaler_X.transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
        y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
        print(
            f"Dataset {dataset_name} preprocessed: {X_train_scaled.shape[0]} train samples, {X_train_scaled.shape[1]} features")
    else:
        # IMDb is pre-processed externally to preserve sparsity structure if needed
        X_train_scaled = X_train
        y_train_scaled = y_train
        X_test_scaled = X_test
        y_test_scaled = y_test
        print(
            f"Dataset {dataset_name} already preprocessed: {X_train_scaled.shape[0]} train samples, {X_train_scaled.shape[1]} features")

    # Constraint Budget Definition
    n_features = X_train_scaled.shape[1]

    # We use Sklearn's Coordinate Descent implementation solely to establish
    # a realistic budget 't' (L1 norm of the unconstrained solution).
    alpha_lasso = 0.1
    print(f"\nComputing reference Lasso (sklearn) with alpha={alpha_lasso}...")
    time_lasso = time.time()
    ref_lasso = Lasso(alpha=alpha_lasso, fit_intercept=False, max_iter=2000, random_state=random_seed)
    ref_lasso.fit(X_train_scaled, y_train_scaled)
    elapsed_time_lasso = time.time() - time_lasso

    lasso_l1_norm = np.linalg.norm(ref_lasso.coef_, 1)
    sparsity_threshold = 1e-6
    lasso_sparsity = np.sum(np.abs(ref_lasso.coef_) > sparsity_threshold)

    print(f"Lasso Reference: t_ref = {lasso_l1_norm:.6f}")
    print(f"Lasso Sparsity: {lasso_sparsity}/{n_features}")
    print(f"Lasso Time: {elapsed_time_lasso:.4f}s")

    if t_value is None:
        t = lasso_l1_norm
        print(f"Using reference t: {t:.6f}")
    else:
        t = t_value
        print(f"Using manual t: {t:.6f}")

    if t != lasso_l1_norm and lasso_l1_norm > 1e-6:
        ratio = (t / lasso_l1_norm) * 100
        print(f"  (Budget is {ratio:.1f}% of reference Lasso norm)")

    # IPM Solver Execution
    print(f"\n[PHASE 1] Running IPM Solver (method={system_method})")
    start_time_ipm = time.time()

    try:
        x_ipm, iter_ipm, lambda_final, mu_final, res_norm_final, debug_info = lasso_ipm_solver_debug(
            X_train_scaled,
            y_train_scaled,
            t,
            tol=1e-8,
            method=system_method,
            cg_maxiter=cg_max_iter,
            verbose=verbosity,
            debug_mode=False,
            max_iter=max_iter
        )
        time_ipm = time.time() - start_time_ipm

        print(f" IPM completed in {time_ipm:.4f}s:")
        print(f"   - Iterations: {iter_ipm}")
        print(f"   - L1 norm: {np.linalg.norm(x_ipm, 1):.6f} / {t:.6f}")
        print(f"   - Sparsity: {np.sum(np.abs(x_ipm) > sparsity_threshold)}/{n_features}")
        ipm_success = True

    except Exception as e:
        print(f" [ERROR] IPM Failed: {str(e)}")
        time_ipm = float('inf')
        iter_ipm = 0
        mu_final = float('inf')
        res_norm_final = float('inf')
        x_ipm = np.zeros(n_features)
        debug_info = {'final_summary': {'total_solver_failures': -1, 'total_numerical_issues': -1,
                                        'convergence_reason': 'FAILED'}}
        ipm_success = False

    # CVXPY Benchmark Execution
    print(f"\n[PHASE 2] Running CVXPY Benchmark...")
    x_cp = cp.Variable(n_features)
    objective = cp.Minimize(0.5 * cp.sum_squares(X_train_scaled @ x_cp - y_train_scaled))
    constraints = [cp.norm(x_cp, 1) <= t]
    problem = cp.Problem(objective, constraints)

    solver_name_cvxpy = "OSQP"
    cvxpy_success = False

    try:
        # Try OSQP (Operator Splitting Quadratic Program) first
        print(f" >> Attempting primary solver ({solver_name_cvxpy})...")
        start_time_cvxpy = time.time()
        problem.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, max_iter=10000, verbose=False)
        time_cvxpy = time.time() - start_time_cvxpy

        # Strict validation of solver status
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise cp.error.SolverError(f"{solver_name_cvxpy} status: {problem.status}")

        cvxpy_success = True
        print(f" >> {solver_name_cvxpy} completed in {time_cvxpy:.4f}s")

    except Exception as e:
        # Fallback to SCS (Splitting Conic Solver) for robustness
        solver_name_cvxpy = "SCS"
        print(f" [FAIL] OSQP failed ({e}). switching to {solver_name_cvxpy}...")
        try:
            start_time_cvxpy = time.time()
            problem.solve(solver=cp.SCS, eps=1e-8, max_iters=10000, verbose=False)
            time_cvxpy = time.time() - start_time_cvxpy
            print(f" >> {solver_name_cvxpy} completed in {time_cvxpy:.4f}s")
            cvxpy_success = True
        except Exception as e2:
            print(f" [ERROR] All CVXPY solvers failed: {e2}")
            time_cvxpy = float('inf')
            cvxpy_success = False

    if not cvxpy_success or x_cp.value is None:
        print("\n [CRITICAL ERROR] Benchmark failed. Skipping comparison.")
        return None

    # Retrieve CVXPY internal statistics
    cvxpy_stats = problem.solver_stats
    time_cvxpy_solve = getattr(cvxpy_stats, 'solve_time', time_cvxpy)
    iterations_cvxpy = getattr(cvxpy_stats, 'num_iters', 0)
    res_pri_cvxpy = getattr(cvxpy_stats, 'res_pri', 0)
    res_dual_cvxpy = getattr(cvxpy_stats, 'res_dual', 0)
    res_norm_cvxpy = max(res_pri_cvxpy, res_dual_cvxpy)

    # Result Aggregation
    results = {
        "dataset": dataset_name,
        "dataset_info": {
            "n_samples": X_train_scaled.shape[0],
            "n_features": n_features,
            "n_test": X_test_scaled.shape[0]
        },
        "problem_setup": {
            "t_budget": t,
            "t_ratio": t / np.sqrt(n_features) if n_features > 0 else 0,
            "reference_l1": t,
            "random_seed": random_seed
        },
        "ipm": {
            "success": ipm_success,
            "time_s": time_ipm,
            "iterations": iter_ipm,
            "final_mu": mu_final if ipm_success else float('inf'),
            "final_res_norm": res_norm_final if ipm_success else float('inf'),
            "l1_norm": np.linalg.norm(x_ipm, 1),
            "l1_utilization": np.linalg.norm(x_ipm, 1) / t if t > 0 else 0,
            "sparsity": np.sum(np.abs(x_ipm) > 1e-6),
            "sparsity_ratio": np.sum(np.abs(x_ipm) > 1e-6) / n_features,
            "mse_train": mean_squared_error(y_train_scaled, X_train_scaled @ x_ipm),
            "mse_test": mean_squared_error(y_test_scaled, X_test_scaled @ x_ipm),
            "mae_test": mean_absolute_error(y_test_scaled, X_test_scaled @ x_ipm),
            "r2_test": r2_score(y_test_scaled, X_test_scaled @ x_ipm),
            "solution_vector": x_ipm,
            "solver_failures": debug_info['final_summary'][
                'total_solver_failures'] if 'final_summary' in debug_info else 0,
            "numerical_issues": debug_info['final_summary'][
                'total_numerical_issues'] if 'final_summary' in debug_info else 0,
            "convergence_reason": debug_info['final_summary'][
                'convergence_reason'] if 'final_summary' in debug_info else 'unknown'
        },
        "cvxpy": {
            "success": cvxpy_success,
            "solver_used": solver_name_cvxpy,
            "time_s": time_cvxpy_solve,
            "iterations": iterations_cvxpy,
            "final_res_norm": res_norm_cvxpy,
            "l1_norm": np.linalg.norm(x_cp.value, 1),
            "l1_utilization": np.linalg.norm(x_cp.value, 1) / t if t > 0 else 0,
            "sparsity": np.sum(np.abs(x_cp.value) > 1e-6),
            "sparsity_ratio": np.sum(np.abs(x_cp.value) > 1e-6) / n_features,
            "mse_train": mean_squared_error(y_train_scaled, X_train_scaled @ x_cp.value),
            "mse_test": mean_squared_error(y_test_scaled, X_test_scaled @ x_cp.value),
            "mae_test": mean_absolute_error(y_test_scaled, X_test_scaled @ x_cp.value),
            "r2_test": r2_score(y_test_scaled, X_test_scaled @ x_cp.value),
            "solution_vector": x_cp.value
        }
    }

    # Comparison Logic
    if ipm_success and cvxpy_success:
        solution_distance = np.linalg.norm(x_ipm - x_cp.value)
        speedup = time_cvxpy_solve / time_ipm if time_ipm > 0 else float('inf')

        results["comparison"] = {
            "solution_distance_l2": solution_distance,
            "speedup_factor": speedup,
            "mse_difference": abs(results["ipm"]["mse_test"] - results["cvxpy"]["mse_test"]),
            "sparsity_agreement": results["ipm"]["sparsity"] == results["cvxpy"]["sparsity"]
        }

        print(f"\n COMPARISON:")
        print(f"   - IPM Speedup: {speedup:.2f}x")
        print(f"   - L2 Distance: {solution_distance:.2e}")
        print(f"   - Sparsity Match: {results['comparison']['sparsity_agreement']}")

    print_compact_report(results, feature_names)
    return results


# --------------------------------------------------------------------------
# EXPERIMENT MANAGEMENT CLASS
# --------------------------------------------------------------------------

class ExperimentRunner:
    """
    Manages the execution flow of the entire project evaluation.
    Handles data ingestion, multiphase experimentation, and plot generation.
    """

    def __init__(self, lasso_ipm_solver_debug, run_experiment_func, verbose=False):
        self.solver = lasso_ipm_solver_debug
        self.run_experiment = run_experiment_func
        self.verbose = verbose
        self.all_results = {}

    def run_convergence_analysis_varying_t(self, dataset_name, data_loader_func, n_t_values=15):
        """
        Phase 1: Sensitivity Analysis.
        Investigates the solver's behavior across a wide range of budgets 't'
        (from highly constrained/sparse to relaxed/dense).
        """
        print(f"\n{'=' * 80}")
        print(f"CONVERGENCE ANALYSIS VARYING t - {dataset_name}")
        print(f"{'=' * 80}")

        X, y, feature_names = data_loader_func()
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Preprocessing consistent with the main experiment
        if dataset_name != 'IMDb':
            scaler_X = StandardScaler().fit(X_train)
            X_train_scaled = scaler_X.transform(X_train)
            scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
            y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
        else:
            X_train_scaled = X_train
            y_train_scaled = y_train

        # Calculate max meaningful t (saturation point)
        lasso_ref = Lasso(alpha=0.01, fit_intercept=False, max_iter=1000)
        lasso_ref.fit(X_train_scaled, y_train_scaled)
        t_max = np.linalg.norm(lasso_ref.coef_, 1)

        if t_max < 1e-6:
            t_max = np.sqrt(X_train_scaled.shape[1])

        t_values = np.logspace(np.log10(0.01 * t_max), np.log10(1.5 * t_max), n_t_values)

        results = {
            'dataset_name': dataset_name,
            't_values': t_values,
            't_max': t_max,
            'shape': X_train_scaled.shape,
            'convergence_data': [],
            'iterations': [],
            'final_mu': [],
            'final_residual': [],
            'solve_times': [],
            'speedup_vs_cvxpy': [],
            'mu_histories': [],
            'comparison_data': []
        }

        print(f"Testing {n_t_values} values of t from {t_values[0]:.4e} to {t_values[-1]:.4e}")
        print(f"Reference t_max = {t_max:.4e}\n")

        for i, t in enumerate(t_values):
            print(f"[{i + 1}/{n_t_values}] Running with t = {t:.4e} ({100 * t / t_max:.1f}% of t_max)")

            # Adaptive Strategy: Switch to CG for large/sparse systems
            if "Synthetic" in dataset_name or "IMDb" in dataset_name or X_train_scaled.shape[1] >= 500:
                method = 'cg'
            else:
                method = 'cholesky'

            try:
                start_time = time.time()
                x_opt, iters, lambda_t, final_mu, final_res, debug_info = self.solver(
                    X_train_scaled, y_train_scaled, t,
                    max_iter=1000,
                    tol=1e-8,
                    verbose=False,
                    debug_mode=True,
                    method=method
                )
                solve_time = time.time() - start_time
                mu_history = debug_info.get('mu_history', [])

                results['convergence_data'].append({
                    't': t,
                    't_normalized': t / t_max,
                    'mu_history': mu_history,
                    'iterations': iters,
                    'solve_time': solve_time
                })
                results['iterations'].append(iters)
                results['final_mu'].append(final_mu)
                results['final_residual'].append(final_res)
                results['solve_times'].append(solve_time)
                results['mu_histories'].append(mu_history)

                print(f"Converged in {iters} iterations, time: {solve_time:.3f}s")

                # Simplified CVXPY check for smaller datasets in the loop
                if X_train_scaled.shape[1] < 2000:
                    x_cp = cp.Variable(X_train_scaled.shape[1])
                    objective = cp.Minimize(0.5 * cp.sum_squares(X_train_scaled @ x_cp - y_train_scaled))
                    constraints = [cp.norm(x_cp, 1) <= t]
                    problem = cp.Problem(objective, constraints)

                    try:
                        start_cvxpy = time.time()
                        problem.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, verbose=False)
                        time_cvxpy = time.time() - start_cvxpy
                        speedup = time_cvxpy / solve_time if solve_time > 0 else 1.0
                        results['speedup_vs_cvxpy'].append(speedup)
                        print(f"  >> CVXPY time: {time_cvxpy:.3f}s, Speedup: {speedup:.2f}x")
                    except:
                        results['speedup_vs_cvxpy'].append(np.nan)
                else:
                    results['speedup_vs_cvxpy'].append(np.nan)

            except Exception as e:
                print(f"  [FAIL] Failed: {str(e)[:50]}...")
                results['convergence_data'].append(None)
                results['iterations'].append(np.nan)
                results['final_mu'].append(np.nan)
                results['final_residual'].append(np.nan)
                results['solve_times'].append(np.nan)
                results['speedup_vs_cvxpy'].append(np.nan)
                results['mu_histories'].append([])

        return results

    def plot_convergence_curves(self, convergence_results, save_path=None):
        # Configuration for Figure 1: Duality Gap History
        n_datasets = len(convergence_results)
        cols = 2
        rows = (n_datasets + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
        axes = axes.flatten()

        for idx, (dataset_name, results) in enumerate(convergence_results.items()):
            ax = axes[idx]
            t_values = results['t_values']
            t_max = results['t_max']
            n_curves = min(7, len(t_values))
            indices = np.linspace(0, len(t_values) - 1, n_curves, dtype=int)
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_curves))

            for i, idx_t in enumerate(indices):
                if results['convergence_data'][idx_t] is not None:
                    mu_history = results['convergence_data'][idx_t]['mu_history']
                    if len(mu_history) > 0:
                        iterations = range(len(mu_history))
                        t_val = t_values[idx_t]
                        ax.semilogy(iterations, np.array(mu_history) + 1e-16,
                                    color=colors[i], linewidth=2, alpha=0.8,
                                    label=f't = {t_val:.2e} ({t_val / t_max * 100:.0f}%)')

            ax.set_xlabel('Iteration')
            ax.set_ylabel('Duality Gap (log scale)')
            ax.set_title(f'{dataset_name} Convergence', fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=1e-8, color='red', linestyle='--', alpha=0.5)

            # Visualization adjustment:
            # Limits x-axis to highlight the active convergence phase.
            ax.set_xlim(left=0, right=15)

        # Hide empty subplots
        for idx in range(n_datasets, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('IPM Convergence Analysis (Zoomed First 30 Iters)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

    def plot_iterations_vs_t(self, convergence_results, save_path=None):
        # Configuration for Figure 2: Performance vs Budget
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax1, ax2 = axes[0], axes[1]

        for dataset_name, results in convergence_results.items():
            t_normalized = results['t_values'] / results['t_max']
            valid_mask = ~np.isnan(results['iterations'])
            if np.any(valid_mask):
                ax1.semilogx(t_normalized[valid_mask], np.array(results['iterations'])[valid_mask],
                             'o-', linewidth=2, markersize=6, alpha=0.8, label=dataset_name)

                if 'speedup_vs_cvxpy' in results:
                    speedups = np.array(results['speedup_vs_cvxpy'])
                    valid_sp = ~np.isnan(speedups)
                    if np.any(valid_sp):
                        ax2.semilogx(t_normalized[valid_sp], speedups[valid_sp],
                                     'o-', linewidth=2, markersize=6, alpha=0.8, label=dataset_name)

        ax1.set_xlabel('Normalized Budget (t/t_max)')
        ax1.set_ylabel('Iterations')
        ax1.set_title('Iterations vs Budget', fontweight='bold')
        ax1.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Set log scale for clearer readability of iteration variance
        ax1.set_yscale('log')
        ax1.set_ylim(bottom=1)


        ax2.axhline(y=1.0, color='black', linestyle='--')
        ax2.set_xlabel('Normalized Budget (t/t_max)')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('Speedup vs CVXPY', fontweight='bold')
        ax2.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        # Set log scale for speedup comparisons
        ax2.set_yscale('log')
        ax2.set_ylim(bottom=0.5)

        plt.suptitle('Performance Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

    def plot_convergence_characterization(self, convergence_results, save_path=None):
        # Configuration for Figure 3: Convergence Rate Analysis
        fig, ax = plt.subplots(figsize=(10, 6))

        counts = {'linear': 0, 'superlinear': 0, 'quadratic': 0}
        for res in convergence_results.values():
            for mu_hist in res['mu_histories']:
                if len(mu_hist) > 5:
                    mu_log = np.log10(np.array(mu_hist[-5:]) + 1e-16)
                    rate = np.mean(np.diff(mu_log))
                    if rate < -1.5:
                        counts['quadratic'] += 1
                    elif rate < -0.8:
                        counts['superlinear'] += 1
                    else:
                        counts['linear'] += 1

        ax.bar(counts.keys(), counts.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_title('Convergence Type Distribution', fontweight='bold')
        ax.set_ylabel('Count')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

    def run_complete_analysis(self, datasets_info):
        """
        Main driver method for the complete analysis pipeline.
        """
        print("=" * 80)
        print("COMPLETE EXPERIMENTAL ANALYSIS - LASSO IPM SOLVER")
        print("=" * 80)

        convergence_results = {}
        comparison_results = {}


        # ================================
        # PHASE 1: VARYING t (Sensitivity)
        # ================================
        print("\n" + "=" * 80)
        print("PHASE 1: CONVERGENCE ANALYSIS VARYING t")
        print("=" * 80)

        for dataset_name, loader_func in datasets_info.items():
            results = self.run_convergence_analysis_varying_t(dataset_name, loader_func, n_t_values=10)
            convergence_results[dataset_name] = results


        # ================================
        # PHASE 2: COMPARISON (Benchmarks)
        # ================================
        print("\n" + "=" * 80)
        print("PHASE 2: DETAILED COMPARISON WITH CVXPY")
        print("=" * 80)

        for dataset_name, loader_func in datasets_info.items():
            print(f"\nRunning comparison for {dataset_name}...")

            # Smart Method Selection [Section 6.1]
            # Use CG for large/sparse/synthetic, Cholesky for small/dense
            if "Synthetic" in dataset_name or "IMDb" in dataset_name:
                method = 'cg'
            else:
                method = 'cholesky'

            comp_result = self.run_experiment(
                dataset_name=dataset_name,
                data_loader_func=loader_func,
                t_value=None,
                system_method=method,
                verbosity=False
            )
            if comp_result:
                comparison_results[dataset_name] = comp_result

        # ==============
        # PHASE 3: PLOTS
        # ==============
        print("\n" + "=" * 80)
        print("PHASE 3: GENERATING PLOTS")
        print("=" * 80)

        import os
        if not os.path.exists('Plots'):
            os.makedirs('Plots')

        self.plot_convergence_curves(convergence_results, 'Plots/convergence_curves_main.png')
        self.plot_iterations_vs_t(convergence_results, 'Plots/performance_vs_t.png')
        self.plot_convergence_characterization(convergence_results, 'Plots/convergence_characterization.png')

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        return convergence_results, comparison_results


# --------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# --------------------------------------------------------------------------
def run_experimental_suite():
    # DATASETS:
    # Diabetes (Small Scale)
    def load_diabetes_data():
        from sklearn.datasets import load_diabetes
        diabetes = load_diabetes()
        return diabetes.data, diabetes.target, diabetes.feature_names

    # California Housing (Medium Scale) ---
    def load_california_data():
        from sklearn.datasets import fetch_california_housing
        california = fetch_california_housing()
        return california.data, california.target, california.feature_names

    # Synthetic High-Dim (Controlled) ---
    def load_synthetic_data():
        print("   Generating Synthetic High-Dim Data (n=2000)...")
        # Params matching the report
        X, y = make_regression(n_samples=3000, n_features=2000, n_informative=9, noise=1.0, random_state=42)
        feature_names = [f"feat_{i}" for i in range(2000)]
        return X, y, feature_names

    # IMDb (Sparse/Large Scale) ---
    def load_imdb_data():
        try:
            df = pd.read_csv('Dataset/imdb_processed.csv')
            if 'target' not in df.columns: return None, None, None
            X = df.drop('target', axis=1).values
            y = df['target'].values
            return X, y, df.drop('target', axis=1).columns.tolist()
        except:
            return None, None, None

    # Register Datasets for the loop
    datasets_info = {
        'Diabetes': load_diabetes_data,
        'California Housing': load_california_data,
        'Synthetic High-Dim': load_synthetic_data
    }

    # Conditional loading for IMDb (in Dataset folder)
    imdb_X, _, _ = load_imdb_data()
    if imdb_X is not None:
        print(">> IMDb dataset found and added.")
        datasets_info['IMDb'] = load_imdb_data
    else:
        print(">> IMDb dataset not found (skipping).")

    # Initialize and Run Suite
    suite = ExperimentRunner(
        lasso_ipm_solver_debug=lasso_ipm_solver_debug,
        run_experiment_func=run_experiment,
        verbose=True
    )
    suite.run_complete_analysis(datasets_info)
    return


if __name__ == "__main__":
    print_environment_info()
    run_experimental_suite()