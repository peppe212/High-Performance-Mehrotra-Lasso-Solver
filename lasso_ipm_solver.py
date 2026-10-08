"""
===============================================================================
A High-Performance Primal-Dual Solver for the Lasso Problem
Implementation of Mehrotra's Predictor-Corrector Interior-Point Method
===============================================================================

Author:         Giuseppe Muschetta
ID:             564026
Program:        M.Sc. in Data Science & Business Informatics
University:     University of Pisa
Course:         Optimization for Data Science
Instructor:     Prof. Antonio Frangioni
Academic Year:  2024/2025
Final Grade:    30/30


CONVENTION USED: Throughout the comments, anything written in square brackets
(e.g., [Sec.], [Eq.], [Algorithm]) refers to specific sections, equations, or
algorithms in the PDF report.

DESCRIPTION:
------------
This module implements a Primal-Dual Path-Following Interior-Point Method
(IPM) to solve the Constrained Lasso problem.

The solver addresses the non-differentiability of the L1-norm via an exact
reformulation into a Quadratic Program (QP) using variable splitting [Sec. 2].
The algorithm employs Mehrotra's Predictor-Corrector scheme [Sec. 3.3] to
achieve Q-superlinear convergence.

Key features consistent with the project report:
- Exact reformulation: x = u - v (u,v >= 0) [Eq. 4-7].
- Normal Equations reduction with low-rank update handling [Sec. 3.4].
- Hybrid Linear Algebra Engine: Automatic switching between Cholesky (direct)
  and Conjugate Gradient (iterative) based on problem scale [Sec. 6.1].
- Adaptive Regularization for handling ill-conditioned KKT systems [Sec. 7.7].
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.sparse.linalg import LinearOperator, cg
import warnings

# Suppressing warnings to keep clean the output log during large benchmarks
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def lasso_ipm_solver_debug(A, b, t,
                           max_iter=1000,
                           tol=1e-8,
                           eta=0.995,
                           method='cholesky',
                           cg_maxiter=1000,
                           cg_rtol=1e-8,
                           initial_delta=1e-8,
                           verbose=True,
                           debug_mode=True):

    """
    Solves the Lasso problem using a Primal-Dual Interior-Point Method.

    Problem Formulation (P) [Eq. 1]:
        min  1/2 * ||Ax - b||_2^2
        s.t. ||x||_1 <= t

    The problem is solved via its QP reformulation [Eq. 4-7]:
        min  1/2 * ||A(u-v) - b||_2^2
        s.t. 1^T(u + v) <= t
             u, v >= 0

    Parameters
    ----------
    A : np.ndarray
        Design matrix of shape (m, n).
    b : np.ndarray
        Target vector of shape (m,).
    t : float
        Constraint budget (L1-norm radius). Must be strictly positive.
    max_iter : int, optional
        Maximum number of IPM iterations (Algorithm 1, line 3).
    tol : float, optional
        Convergence tolerance for duality gap and feasibility residuals.
    eta : float, optional
        Step-size scaling factor to ensure strict feasibility (fraction-to-boundary).
    method : {'cholesky', 'cg'}, optional
        Linear system solver strategy [Sec. 6.1].
        'cholesky' for direct factorization (dense/small n).
        'cg' for matrix-free iterative solution (large n).
    initial_delta : float, optional
        Initial regularization parameter for the adaptive stability mechanism.
    verbose : bool, optional
        Enables iteration logging.
    debug_mode : bool, optional
        Collects extended telemetry for experimental analysis (Chapter 7).

    Returns
    -------
    x_star : np.ndarray
        Optimal solution vector (u - v).
    iter_count : int
        Number of iterations to convergence.
    lambda_t : float
        Optimal dual multiplier associated with the budget constraint.
    mu : float
        Final duality measure.
    res_norm : float
        Final KKT residual norm (infinity norm).
    debug_info : dict
        Dictionary containing convergence history and stability metrics.
    """

    # --------------------------------------------------------------------------
    # PRE-PROCESSING PHASE [Sec. 5.1]
    # --------------------------------------------------------------------------

    # Input validation
    if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray):
        raise ValueError("A and b must be numpy arrays")
    if A.ndim != 2 or b.ndim != 1:
        raise ValueError("A must be a 2D matrix, b must be a 1D vector")
    if A.shape[0] != b.shape[0]:
        raise ValueError("Number of rows in A must match the length of b")
    if t <= 0:
        raise ValueError("Parameter t must be positive")
    if not (0 < eta < 1):
        raise ValueError("Parameter eta must be between 0 and 1")

    # Initialize debugging data structures for algorithm diagnostics
    debug_info = {
        'iterations': [],
        'condition_numbers': [],
        'regularization_history': [],
        'step_sizes': [],
        'mu_history': [],
        'residual_history': [],
        'solver_failures': [],
        'numerical_issues': []
    }

    # Problem dimensions and numerical constants
    m, n = A.shape
    eps = 1e-12
    k = 0

    if debug_mode:
        print(f"DEBUG: Problem dimension ({m}, {n}), budget t={t:.4e}")

    # ----------------------------------------------------------------------------
    # DESIGN CHOICE: PURE HESSIAN & DYNAMIC REGULARIZATION
    # ----------------------------------------------------------------------------
    # We maintain the Hessian G_qp based on the pure A^T @ A [Sec. 5.1].
    # Rank deficiencies and ill-conditioning are handled exclusively via
    # transient Tikhonov regularization (delta) inside the linear solver [Sec. 6.1].
    # ----------------------------------------------------------------------------

    AtA_base = A.T @ A
    original_rank = np.linalg.matrix_rank(A, tol=1e-12)
    cond_number = np.linalg.cond(AtA_base)
    debug_info['condition_numbers'].append(cond_number)

    if debug_mode:
        print(f"DEBUG: Conditioning AtA = {cond_number:.2e}, Rank = {original_rank}/{min(m, n)}")

    # Determine if we need aggressive regularization (stored for later use)
    needs_regularization = (cond_number > 1e10) or (original_rank < n) # boolean variable
    base_reg_factor = 0.0

    if needs_regularization:
        if cond_number > 1e10:
            base_reg_factor = max(1e-12, 1e-14 * cond_number * np.sqrt(m))
        elif original_rank < n:
            base_reg_factor = 1e-10

        if debug_mode:
            print(f"DEBUG: Conditioning issue detected, base regularization = {base_reg_factor:.2e}")


    # --------------------------------------------------------------------------
    # INITIALIZATION STRATEGY [Sec. 3.1]
    # --------------------------------------------------------------------------

    # Constructing a strictly feasible starting point w0.
    scale_factor = np.sqrt(np.mean(np.sum(A ** 2, axis=0)))
    alpha_p = min(t / (4 * n), max(1e-6, t / scale_factor))
    u, v = np.full(n, alpha_p), np.full(n, alpha_p)
    s_t = max(tol * 100, t - np.sum(u + v))

    # Initial dual variables scaled for conditioning
    dual_scale = min(10.0, max(0.1, 1.0 / np.sqrt(cond_number) * 1000))
    zu, zv = np.ones(n) * dual_scale, np.ones(n) * dual_scale
    lambda_t = dual_scale

    if debug_mode:
        print(f"DEBUG: Adaptive initialization - scale={scale_factor:.2e}, dual_scale={dual_scale:.2e}")
        print(f"DEBUG: Initialization - alpha_p={alpha_p:.2e}, s_t={s_t:.2e}")

    # Pre-computations for efficiency
    diag_AtA_base = AtA_base.diagonal().copy()
    # Explicit construction of the full Hessian G_qp [Sec. 2]
    G_qp = np.block([[AtA_base, -AtA_base], [-AtA_base, AtA_base]])
    e_2n = np.ones(2 * n)
    delta = initial_delta
    stall_counter = 0
    consecutive_small_steps = 0


    # ==========================================
    # MAIN IPM LOOP (Algorithm, report page 9)
    # ==========================================
    for k in range(max_iter):
        iter_info = {'iteration': k, 'warnings': []}

        # --------------------------------------------------------------------------
        # NUMERICAL SAFETY: DUAL VARIABLE CLAMPING
        # --------------------------------------------------------------------------
        # We constrain dual variables to a safe range [1e-12, 1e12] to prevent
        # floating-point overflows in the complementarity products (u*zu).
        # --------------------------------------------------------------------------

        zu = np.clip(zu, 1e-12, 1e12)
        zv = np.clip(zv, 1e-12, 1e12)
        lambda_t = np.clip(lambda_t, 1e-12, 1e12)

        # Compute residuals for feasibility [Algorithm, Line 5]
        # Based on KKT conditions Eq. (9-11)
        x_k = u - v
        r_ls = A @ x_k - b
        r_d1 = A.T @ r_ls - zu + lambda_t
        r_d2 = -(A.T @ r_ls) - zv + lambda_t
        r_p = np.sum(u + v) + s_t - t

        # Compute KKT residual norm and Duality Measure [Algorithm, Line 6]
        res_norm = max(np.linalg.norm(r_d1, np.inf), np.linalg.norm(r_d2, np.inf), np.abs(r_p))
        rc_u, rc_v, rc_t = u * zu, v * zv, s_t * lambda_t
        # Duality Measure is the normalized form of Duality Gap
        mu = (np.sum(rc_u) + np.sum(rc_v) + rc_t) / (2 * n + 1)

        # DEBUGGING: Check for numerical issues
        if debug_mode:
            # Check for NaN/Inf in variables
            vars_to_check = [u, v, zu, zv, lambda_t, s_t]
            var_names = ['u', 'v', 'zu', 'zv', 'lambda_t', 's_t']
            for var, name in zip(vars_to_check, var_names):
                if np.any(np.isnan(var)) or np.any(np.isinf(var)):
                    issue = f"NaN/Inf in {name}"
                    iter_info['warnings'].append(issue)
                    debug_info['numerical_issues'].append(f"Iter {k}: {issue}")

            # Check complementarity violations
            comp_violations = np.sum((u < 1e-10) & (zu > 1e10)) + np.sum((v < 1e-10) & (zv > 1e10))
            if comp_violations > 0:
                iter_info['warnings'].append(f"Complementarity violations: {comp_violations}")

        # Store convergence metrics
        debug_info['mu_history'].append(mu)
        debug_info['residual_history'].append(res_norm)

        if verbose:
            warning_str = f" {len(iter_info['warnings'])} warnings" if iter_info['warnings'] else ""
            print(f"--- Iter {k:02d} --- Mu = {mu:.4e}, ResNorm = {res_norm:.4e}, Delta = {delta:.1e}{warning_str}")
            if debug_mode and iter_info['warnings']:
                for w in iter_info['warnings']:
                    print(f"    {w}")


        # --------------------------------------------------------------------------
        # TERMINATION CRITERIA [Algorithm, Lines 7-9]
        # --------------------------------------------------------------------------
        # Convergence is declared when both Primal/Dual residuals and the Duality
        # Gap (mu) fall below the established tolerance.
        # --------------------------------------------------------------------------

        if mu < tol and res_norm < tol * 100:
            if verbose:
                print(f"\nCONVERGENCE REACHED in {k + 1} iterations.")
                print(f"    Final mu={mu:.2e}, res_norm={res_norm:.2e}")
            debug_info['iterations'] = list(range(k + 1))
            debug_info['final_summary'] = {
                'total_iterations': k + 1,
                'total_solver_failures': len(debug_info['solver_failures']),
                'total_numerical_issues': len(debug_info['numerical_issues']),
                'final_condition': cond_number,
                'convergence_reason': 'tolerance'
            }
            return u - v, k + 1, lambda_t, mu, res_norm, debug_info

        # System assembly and solving with enhanced diagnostics
        # REDUCTION TO NORMAL EQUATIONS [Sec. 3.4, Eq. 16 solving for Dk]
        diag_Dk_vec = np.concatenate([zu / (u + eps), zv / (v + eps)])
        lambda_s_ratio = lambda_t / (s_t + eps)

        def solve_linear_system_debug(rhs, step_name="unknown"):
            """
            Robust linear system solver with automatic regularization adjustment.
            Implements the Core Solver Engine described in [Sec. 6.1].
            """
            nonlocal delta

            for attempt in range(10):

                if method == 'cholesky':

                    # ----------------------------------------------------------
                    # DIRECT METHOD (Cholesky O(n^3)) [Sec. 5.2]
                    # ----------------------------------------------------------
                    # Reduced system construction with low-rank update handling.
                    # M = G_qp + D_k + lambda/s * 11^T + delta * I
                    # ----------------------------------------------------------

                    # Start with the base system
                    M = G_qp + np.diag(diag_Dk_vec) + lambda_s_ratio * np.outer(e_2n, e_2n)

                    # Apply regularization if needed
                    current_reg = max(delta, base_reg_factor)
                    M = M + current_reg * np.eye(2 * n)

                    try:
                        # Check conditioning before proceeding
                        if debug_mode and attempt == 0:
                            cond_M = np.linalg.cond(M)
                            if cond_M > 1e13:
                                iter_info['warnings'].append(f"Matrix condition {cond_M:.1e} in {step_name}")
                                if debug_mode:
                                    print(f"    Critical conditioning {cond_M:.1e} in {step_name}")

                        # Attempt Cholesky factorization
                        L, lower = cho_factor(M, lower=False, check_finite=False)
                        solution = cho_solve((L, lower), rhs)

                        # Verify solution quality
                        if np.any(np.isnan(solution)) or np.any(np.isinf(solution)):
                            if debug_mode:
                                print(f"    NaN/Inf solution in {step_name} (attempt {attempt + 1})")
                            debug_info['solver_failures'].append(
                                f"Iter {k}, {step_name}, attempt {attempt + 1}: NaN/Inf solution")
                            delta *= 10
                            continue

                        # Check solution norm (sanity check)
                        sol_norm = np.linalg.norm(solution)
                        if sol_norm > 1e8:
                            if debug_mode:
                                print(f"    Unstable solution ||sol||={sol_norm:.1e} in {step_name}")
                            delta *= 5
                            continue

                        # Success!
                        if debug_mode and attempt > 0:
                            print(f"    Cholesky success on attempt {attempt + 1} with delta={delta:.2e}")

                        return solution

                    except (LinAlgError, ValueError) as e:
                        if debug_mode:
                            error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
                            print(f"    Cholesky failed (attempt {attempt + 1}): {error_msg}")
                        debug_info['solver_failures'].append(
                            f"Iter {k}, {step_name}, attempt {attempt + 1}: {type(e).__name__}")
                        delta *= 10

                elif method == 'cg':

                    # --------------------------------------------------------------------
                    # ITERATIVE METHOD (Conjugate Gradient O(n^2)) [Sec. 6.1]
                    # --------------------------------------------------------------------
                    # Matrix-Free implementation for high-dimensional instances.
                    # --------------------------------------------------------------------

                    current_reg = max(delta, base_reg_factor)

                    def matvec(p):
                        """Matrix-vector product for the KKT system."""
                        p_u, p_v = p[:n], p[n:]
                        common_term = AtA_base @ (p_u - p_v)
                        res_g = np.concatenate([common_term, -common_term])
                        res_d = diag_Dk_vec * p
                        res_rank1 = (lambda_s_ratio * e_2n.dot(p)) * e_2n
                        res_delta = current_reg * p
                        return res_g + res_d + res_rank1 + res_delta

                    M_op = LinearOperator((2 * n, 2 * n), matvec=matvec)

                    # More robust preconditioner
                    precond_vec = 1.0 / (diag_AtA_base.repeat(2) + diag_Dk_vec + current_reg + lambda_s_ratio + eps)
                    precond_vec = np.clip(precond_vec, 1e-10, 1e10)
                    precond = LinearOperator((2 * n, 2 * n), matvec=lambda p: p * precond_vec)

                    try:
                        solution, info = cg(M_op, rhs, rtol=cg_rtol, maxiter=cg_maxiter, M=precond)

                        if info == 0:
                            # Verify CG solution quality
                            if np.any(np.isnan(solution)) or np.any(np.isinf(solution)):
                                if debug_mode:
                                    print(f"    CG solution NaN/Inf in {step_name}")
                                delta *= 10
                                continue

                            sol_norm = np.linalg.norm(solution)
                            if sol_norm > 1e8:
                                if debug_mode:
                                    print(f"    CG unstable solution ||sol||={sol_norm:.1e}")
                                delta *= 5
                                continue

                            if debug_mode and attempt > 0:
                                print(f"    CG success on attempt {attempt + 1} with delta={delta:.2e}")
                            return solution
                        else:
                            if debug_mode:
                                cg_status = {0: "success", 1: "max_iter", -1: "illegal_input", -2: "breakdown"}
                                status_msg = cg_status.get(info, f'code_{info}')
                                print(f"    CG failed (attempt {attempt + 1}): {status_msg}")
                            debug_info['solver_failures'].append(f"Iter {k}, {step_name}, CG code {info}")
                            delta *= 10

                    except Exception as e:
                        if debug_mode:
                            print(f"    CG exception (attempt {attempt + 1}): {str(e)[:30]}...")
                        debug_info['solver_failures'].append(f"Iter {k}, {step_name}, CG exception")
                        delta *= 10
                else:
                    raise ValueError("Parameter 'method' must be 'cholesky' or 'cg'")

            # If we get here, all attempts have failed
            debug_info['solver_failures'].append(f"Iter {k}, {step_name}: PERSISTENT FAILURE after 10 attempts")
            if debug_mode:
                print(f"    PERSISTENT ERROR in {step_name} after 10 attempts (final delta: {delta:.2e})")

            return None


        # ======================================================================
        # STEP 1: PREDICTOR (Affine-Scaling Direction) [Algorithm, Lines 10-14]
        # ======================================================================
        # Solving the linear system for mu=0 to find the affine direction.

        rhs_comp_t_aff = -rc_t
        correction_from_t_aff = (rhs_comp_t_aff + lambda_t * r_p) / (s_t + eps)
        rhs_u_aff = -r_d1 - zu - correction_from_t_aff
        rhs_v_aff = -r_d2 - zv - correction_from_t_aff
        d_uv_aff = solve_linear_system_debug(np.concatenate([rhs_u_aff, rhs_v_aff]), "predictor")

        if d_uv_aff is None:
            if verbose:
                print("CRITICAL ERROR: Linear solver persistently failed (predictor).")
            break

        # Extract affine scaling directions
        du_aff, dv_aff = d_uv_aff[:n], d_uv_aff[n:]
        ds_t_aff = -r_p - np.sum(du_aff + dv_aff)
        dzu_aff = (-rc_u - zu * du_aff) / (u + eps)
        dzv_aff = (-rc_v - zv * dv_aff) / (v + eps)
        dlambda_t_aff = (-rc_t - lambda_t * ds_t_aff) / (s_t + eps)

        # Step size computation
        alpha_pri_aff = min(1.0, *([-v / dv for v, dv in
                                    zip(np.concatenate([u, v, [s_t]]), np.concatenate([du_aff, dv_aff, [ds_t_aff]]))
                                    if dv < -tol] or [1.0]))
        alpha_dual_aff = min(1.0, *([-v / dv for v, dv in
                                     zip(np.concatenate([zu, zv, [lambda_t]]),
                                         np.concatenate([dzu_aff, dzv_aff, [dlambda_t_aff]]))
                                     if dv < -tol] or [1.0]))

        # Compute centering parameter sigma [Mehrotra's Heuristic, Sec. 3.3]
        mu_aff_num = ((u + alpha_pri_aff * du_aff).dot(zu + alpha_dual_aff * dzu_aff) +
                      (v + alpha_pri_aff * dv_aff).dot(zv + alpha_dual_aff * dzv_aff) +
                      (s_t + alpha_pri_aff * ds_t_aff) * (lambda_t + alpha_dual_aff * dlambda_t_aff))
        mu_aff = max(0, mu_aff_num / (2 * n + 1))

        if np.isnan(mu_aff) or np.isinf(mu_aff) or mu_aff < 0:
            sigma = 0.1
            if debug_mode:
                iter_info['warnings'].append(f"Invalid mu_aff={mu_aff}, using sigma=0.1")
        else:
            sigma = (mu_aff / mu) ** 3 if mu > eps else 0.0

        # ======================================================================
        # STEP 2: CORRECTOR (Centering + Correction) [Algorithm, Lines 15-18]
        # ======================================================================
        # Incorporates the centering parameter and second-order corrections.
        # This step ensures Q-Superlinear convergence [Sec. 4.2].

        rc_corr_u, rc_corr_v, rc_corr_t = du_aff * dzu_aff, dv_aff * dzv_aff, ds_t_aff * dlambda_t_aff
        rhs_comp_t_corr = sigma * mu - rc_t - rc_corr_t
        correction_from_t_corr = (rhs_comp_t_corr + lambda_t * r_p) / (s_t + eps)
        rhs_u_corr = -r_d1 - (rc_u + rc_corr_u - sigma * mu) / (u + eps) - correction_from_t_corr
        rhs_v_corr = -r_d2 - (rc_v + rc_corr_v - sigma * mu) / (v + eps) - correction_from_t_corr
        d_uv = solve_linear_system_debug(np.concatenate([rhs_u_corr, rhs_v_corr]), "corrector")

        if d_uv is None:
            if verbose:
                print("CRITICAL ERROR: Linear solver persistently failed (corrector).")
            break

        # Extract final search directions
        du, dv = d_uv[:n], d_uv[n:]
        ds_t = -r_p - np.sum(du + dv)
        dzu = (-rc_u - rc_corr_u + sigma * mu - zu * du) / (u + eps)
        dzv = (-rc_v - rc_corr_v + sigma * mu - zv * dv) / (v + eps)
        dlambda_t = (-rc_t - rc_corr_t + sigma * mu - lambda_t * ds_t) / (s_t + eps)

        # ----------------------------------------------------------------------
        # LINE SEARCH: Fraction-to-the-Boundary [Algorithm, Lines 19-20]
        # ----------------------------------------------------------------------
        # Scaling step sizes by eta to stay strictly feasible.
        # ----------------------------------------------------------------------

        alpha_pri_max = min(1.0, *([-v / dv for v, dv in
                                    zip(np.concatenate([u, v, [s_t]]), np.concatenate([du, dv, [ds_t]]))
                                    if dv < -tol] or [1.0]))
        alpha_dual_max = min(1.0, *([-v / dv for v, dv in
                                     zip(np.concatenate([zu, zv, [lambda_t]]), np.concatenate([dzu, dzv, [dlambda_t]]))
                                     if dv < -tol] or [1.0]))
        alpha_pri, alpha_dual = eta * alpha_pri_max, eta * alpha_dual_max

        debug_info['step_sizes'].append((alpha_pri, alpha_dual))

        # Enhanced stall detection
        if alpha_pri < tol and alpha_dual < tol:
            stall_counter += 1
            consecutive_small_steps += 1
        else:
            stall_counter = 0
            consecutive_small_steps = 0 if max(alpha_pri, alpha_dual) > 0.01 else consecutive_small_steps + 1

        if stall_counter >= 3 or consecutive_small_steps >= 5:
            if verbose:
                reason = "numerical stall" if stall_counter >= 3 else "consecutive small steps"
                print(f"\nCONVERGENCE REACHED ({reason}).")
            debug_info['final_summary'] = {
                'total_iterations': k + 1,
                'total_solver_failures': len(debug_info['solver_failures']),
                'total_numerical_issues': len(debug_info['numerical_issues']),
                'final_condition': cond_number,
                'convergence_reason': 'stall',
            }
            break

        # Variable updates [Algorithm, Lines 21-22]
        u, v, s_t = u + alpha_pri * du, v + alpha_pri * dv, s_t + alpha_pri * ds_t
        zu, zv, lambda_t = zu + alpha_dual * dzu, zv + alpha_dual * dzv, lambda_t + alpha_dual * dlambda_t

        # Adaptive delta reduction [Sec. 7.7]
        delta = max(initial_delta, delta / 1.5)
        debug_info['regularization_history'].append(delta)

        # Store iteration information
        iter_info['delta'] = delta
        iter_info['mu'] = mu
        iter_info['res_norm'] = res_norm
        iter_info['alpha'] = (alpha_pri, alpha_dual)
        debug_info['iterations'].append(iter_info)

    # Final reporting
    final_iter_count = k + 1
    if k == max_iter - 1:
        print(f"\nWARNING: Maximum number of iterations ({max_iter}) reached.")

    debug_info['final_summary'] = {
        'total_iterations': final_iter_count,
        'total_solver_failures': len(debug_info['solver_failures']),
        'total_numerical_issues': len(debug_info['numerical_issues']),
        'final_condition': cond_number,
        'convergence_reason': 'max_iter' if k == max_iter - 1 else 'tolerance'
    }

    return u - v, final_iter_count, lambda_t, mu, res_norm, debug_info
