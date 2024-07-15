import time
import numpy as np

from scipy.stats import qmc
from scipy.optimize import minimize


class G_DFL:

    def __init__(self, 
                 verbose, save_logs, 
                 max_time, max_fun, max_it, max_num_disceret_dirs, 
                 tolerance_for_best,
                 tolerance_for_continuous_dir, armijo_alpha_0, armijo_gamma, armijo_delta, armijo_min_alpha,
                 eta_for_discrete_dir, xi_for_discrete_dir, min_xi_for_discrete_dir):
        
        self.verbose = verbose
        self.save_logs = save_logs

        self.max_time = max_time
        self.max_fun = max_fun
        self.max_it = max_it
        self.max_num_discret_dirs = max_num_disceret_dirs

        self.tolerance_for_best = tolerance_for_best

        self.tolerance_for_continuous_dir = tolerance_for_continuous_dir
        self.armijo_alpha_0 = armijo_alpha_0
        self.armijo_gamma = armijo_gamma
        self.armijo_delta = armijo_delta
        self.armijo_min_alpha = armijo_min_alpha

        self.eta_for_discrete_dir = eta_for_discrete_dir
        self.xi_for_discrete_dir = xi_for_discrete_dir
        self.min_xi_for_discrete_dir = min_xi_for_discrete_dir

    def cs_search(self, x, f, type_gradient_related_direction, max_continuous_iter, problem, n_f_evals, n_g_evals, start_time):
        
        alpha_fw = 0
        cont_moved = False

        if type_gradient_related_direction == 'lbfgs':

            res = minimize(fun=problem.feval,
                           x0=x,
                           method='L-BFGS-B',
                           jac=problem.geval,
                           bounds=[(problem.lb[i], problem.ub[i]) for i in range(problem.n)],
                           options={'maxiter': max_continuous_iter, 'maxfun': self.max_fun - n_f_evals, 'gtol': self.tolerance_for_continuous_dir})

            n_f_evals += res.nfev
            n_g_evals += res.njev

            if n_f_evals >= self.max_fun or time.time() - start_time > self.max_time:
                return x, f, 0, False, n_f_evals, n_g_evals

            if res.status in [0, 1] and not np.allclose(x, res.x):
                y = res.x
                fy = res.fun
                alpha_fw = 1
                cont_moved = True

            else:
                y = x
                fy = f
                alpha_fw = 0

        elif type_gradient_related_direction in ['wolfe', 'pj']:

            x_tmp = np.copy(x)
            f_tmp = f

            for _ in range(max_continuous_iter):

                gr = problem.geval(x_tmp)
                n_g_evals += 1

                if n_f_evals >= self.max_fun or time.time() - start_time > self.max_time:
                    return x, f, 0, False, n_f_evals, n_g_evals

                if np.linalg.norm(np.maximum(problem.lb, np.minimum(x_tmp - gr, problem.ub)) - x_tmp, ord=np.inf) <= self.tolerance_for_continuous_dir:
                    alpha_fw = 0
                    break

                gr[np.where(np.abs(gr) <= self.tolerance_for_continuous_dir)[0]] = 0
                dir = np.zeros(problem.n)

                if type_gradient_related_direction == 'wolfe':
                    gr_0_idx = np.where(gr > 0)[0]
                    dir[gr_0_idx] = problem.lb[gr_0_idx] - x_tmp[gr_0_idx]

                    ls_0_idx = np.where(gr < 0)[0]
                    dir[ls_0_idx] = problem.ub[ls_0_idx] - x_tmp[ls_0_idx]

                else:
                    gr_x_lb_idx = np.where(gr > x_tmp - problem.lb)[0]
                    dir[gr_x_lb_idx] = problem.lb[gr_x_lb_idx] - x_tmp[gr_x_lb_idx]

                    ls_x_ub_idx = np.where(gr < x_tmp - problem.ub)[0]
                    dir[ls_x_ub_idx] = problem.ub[ls_x_ub_idx] - x_tmp[ls_x_ub_idx]

                    others_idx = np.setdiff1d(np.setdiff1d(np.arange(problem.n), gr_x_lb_idx), ls_x_ub_idx)
                    dir[others_idx] = -gr[others_idx]

                alpha_fw = self.armijo_alpha_0

                f_tmp_2 = problem.feval(x_tmp + alpha_fw * dir)
                n_f_evals += 1

                if n_f_evals >= self.max_fun or time.time() - start_time > self.max_time:
                    return x, f, 0, False, n_f_evals, n_g_evals

                while f_tmp_2 > f_tmp + alpha_fw * self.armijo_gamma * np.dot(gr, dir) and alpha_fw > self.armijo_min_alpha:
                    alpha_fw *= self.armijo_delta

                    f_tmp_2 = problem.feval(x_tmp + alpha_fw * dir)
                    n_f_evals += 1

                    if n_f_evals >= self.max_fun or time.time() - start_time > self.max_time:
                        return x, f, 0, False, n_f_evals, n_g_evals

                if alpha_fw > self.armijo_min_alpha:
                    x_tmp = x_tmp + alpha_fw * dir
                    f_tmp = f_tmp_2
                    cont_moved = True
                else:
                    alpha_fw = 0
                    break

            y = x_tmp
            fy = f_tmp

        else:
            raise AssertionError('Gradient-related direction unknown')
        
        return y, fy, alpha_fw, cont_moved, n_f_evals, n_g_evals
    
    def nm_discrete_linesearch(self, y, fy, d, alpha_tilde, xi, problem, n_f_evals):
        n = len(d)

        alpha_max = np.inf * np.ones(n)

        indices = (d > 0)
        alpha_max[indices] = np.divide(problem.ub[indices] - y[indices], d[indices])
        indices = (d < 0)
        alpha_max[indices] = np.divide(problem.lb[indices] - y[indices], d[indices])

        alpha_bar = np.floor(np.min(alpha_max))
        alpha_init = min(alpha_tilde, alpha_bar)

        if alpha_init > 0:
            y_trial = y + alpha_init * d

            f_trial = problem.feval(y_trial)
            n_f_evals += 1

            if n_f_evals >= self.max_fun:
                return 0, y, np.inf, n_f_evals

        else:
            f_trial = np.inf

        if alpha_init > 0 and f_trial <= fy - xi:

            alpha = alpha_init
            x = y_trial
            f = f_trial

            if alpha < alpha_bar:
                y_trial = y + min(alpha_bar, 2 * alpha) * d

                f_trial = problem.feval(y_trial)
                n_f_evals += 1

                if n_f_evals >= self.max_fun:
                    return 0, y, np.inf, n_f_evals

            else:
                f_trial = np.inf

            while alpha < alpha_bar and f_trial <= fy - xi:

                alpha = min(alpha_bar, 2 * alpha)

                x = y_trial
                f = f_trial

                if alpha < alpha_bar:
                    y_trial = y + min(alpha_bar, 2 * alpha) * d

                    f_trial = problem.feval(y_trial)
                    n_f_evals += 1

                    if n_f_evals >= self.max_fun:
                        return 0, y, np.inf, n_f_evals

                else:
                    f_trial = np.inf

        else:
            alpha = 0
            x = y
            f = np.inf

        return alpha, x, f, n_f_evals
    
    @staticmethod
    def prime_vector(d):
        
        n_int = len(d)

        if n_int == 1:
            return True

        temp = np.gcd(np.array(abs(d[0]), dtype=int), np.array(abs(d[1]), dtype=int))
        if n_int == 2:
            return temp == 1

        for i in np.arange(2, n_int, 1):
            temp = np.gcd(temp, np.array(abs(d[i]), dtype=int))

            if temp == 1:
                return True

        if temp != 1:
            return False
    
    def generate_dirs(self, n_cont, n_int, D, succ, alpha_tilde, eta_for_dir, sequencer):

        for _ in range(1000):

            v = 2 * sequencer.random(1) - np.ones(n_int)
            v = eta_for_dir * (v / np.linalg.norm(v))

            if np.linalg.norm(v) < 1e-16:
                break

            d = np.round(v)

            if self.prime_vector(d):

                d = np.reshape(d, (d.shape[1], 1))
                d = np.concatenate((np.zeros((n_cont, 1)), d), axis=0)

                diff1 = D - np.tile(d, (1, np.shape(D)[1]))
                diff2 = D + np.tile(d, (1, np.shape(D)[1]))

                if not ((np.min(np.sum(np.abs(diff1), axis=0)) == 0) or (np.min(np.sum(np.abs(diff2), axis=0)) == 0)):
                    Dout = np.hstack((d.copy(), D))
                    succ_out = np.hstack((np.array(0), succ))
                    alpha = np.hstack((np.array(max(0, np.max(alpha_tilde))), alpha_tilde))

                    return Dout, succ_out, alpha, 1

        Dout = D
        succ_out = succ
        alpha = alpha_tilde

        return Dout, succ_out, alpha, 0

    def run(self, seed, type_gradient_related_direction, max_continuous_iter, problem, start_time):

        if problem.nint <= 1:
            raise AssertionError("ERROR: number of integer variables must be > 1.")

        ub_int = np.copy(problem.ub)
        lb_int = np.copy(problem.lb)
        ub_int[problem.ncont:] = np.round(ub_int[problem.ncont:])
        lb_int[problem.ncont:] = np.round(lb_int[problem.ncont:])

        if np.sum(np.abs(problem.ub[problem.ncont:] - ub_int[problem.ncont:])) != 0 or np.sum(np.abs(problem.lb[problem.ncont:] - lb_int[problem.ncont:])) != 0:
            raise AssertionError("ERROR: upper and/or lower bound on some variable is NOT integer.")

        if np.min((problem.x_initial >= problem.lb) & (problem.x_initial <= problem.ub)) == 0:
            raise RuntimeError("ERROR: Initial point does not satisfy the bound constraints.")

        n_f_evals = 0
        n_g_evals = 0
        n_iter = 0
        all_ones_for_ds = 0
        xi = self.xi_for_discrete_dir
        eta_for_dir = self.eta_for_discrete_dir

        sequencer = qmc.Halton(d=problem.nint, scramble=True, seed=seed)

        x = np.copy(problem.x_initial)
        x[problem.ncont:] = np.round(x[problem.ncont:])
        f = problem.feval(x)
        n_f_evals += 1

        best_f = f
        best_x = x
        best_nf = n_f_evals
        best_ng = n_g_evals
        best_it = n_iter
        best_time = time.time() - start_time

        if n_f_evals >= self.max_fun:
            return best_x, best_f, "Max function evaluations reached", n_f_evals, best_nf, n_iter, best_it, best_time, n_g_evals, best_ng

        D = np.concatenate((np.zeros((problem.ncont, problem.nint)), np.identity(problem.nint)), axis=0)
        succ = np.zeros(problem.nint)

        alpha_tilde = np.round((ub_int[problem.ncont:] - lb_int[problem.ncont:]) / 2.0)
        old_max_alpha = np.inf

        if self.verbose:
            print_format = '%s|   %5d |        %5d | %+13.8e | %+13.8e | %5d/%5d |   '
            print(' T |  n_iter |    n_f_evals |        f     |    max_alpha    |        ndir |')

        cont_moved = True
        int_moved = True

        while True:

            n_iter += 1

            if n_iter > self.max_it:
                return best_x, best_f, "Max iterations reached", n_f_evals, best_nf, n_iter - 1, best_it, best_time, n_g_evals, best_ng

            if time.time() - start_time > self.max_time:
                return best_x, best_f, "Max time reached", n_f_evals, best_nf, n_iter - 1, best_it, best_time, n_g_evals, best_ng

            if not cont_moved and not int_moved:
                y = x
                fy = f

            else:                
                y, fy, alpha_fw, cont_moved, n_f_evals, n_g_evals = self.cs_search(x, f, type_gradient_related_direction, max_continuous_iter, problem, n_f_evals, n_g_evals, start_time) 
                
                if n_f_evals > self.max_fun:
                    return best_x, best_f, "Max function evaluations reached", n_f_evals, best_nf, n_iter - 1, best_it, best_time, n_g_evals, best_ng

                if time.time() - start_time > self.max_time:
                    return best_x, best_f, "Max time reached", n_f_evals, best_nf, n_iter - 1, best_it, best_time, n_g_evals, best_ng

                if self.verbose:
                    print(print_format % ('(c)', n_iter, n_f_evals, fy, alpha_fw, 1, 1))
            
            if fy < best_f - self.tolerance_for_best:
                best_x = y
                best_f = fy
                best_nf = n_f_evals
                best_ng = n_g_evals
                best_it = n_iter
                best_time = time.time() - start_time

            if cont_moved:
                all_ones_for_ds = 0

            int_moved = False

            for i_dir in range(D.shape[1]):

                if time.time() - start_time > self.max_time:
                    return best_x, best_f, "Max time reached", n_f_evals, best_nf, n_iter, best_it, best_time, n_g_evals, best_ng

                d = D[:, i_dir]

                f_ref = fy

                alpha, x_trial, f_trial, n_f_evals = self.nm_discrete_linesearch(y, fy, d, alpha_tilde[i_dir], xi, problem, n_f_evals)

                if n_f_evals >= self.max_fun:
                    return best_x, best_f, "Max function evaluations reached", n_f_evals, best_nf, n_iter, best_it, best_time, n_g_evals, best_ng

                if time.time() - start_time > self.max_time:
                    return best_x, best_f, "Max time reached", n_f_evals, best_nf, n_iter, best_it, best_time, n_g_evals, best_ng

                if alpha <= 0:
                    d = -d

                    alpha, x_trial, f_trial, n_f_evals = self.nm_discrete_linesearch(y, fy, d, alpha_tilde[i_dir], xi, problem, n_f_evals)

                    if n_f_evals >= self.max_fun:
                        return best_x, best_f, "Max function evaluations reached", n_f_evals, best_nf, n_iter, best_it, best_time, n_g_evals, best_ng

                    if time.time() - start_time > self.max_time:
                        return best_x, best_f, "Max time reached", n_f_evals, best_nf, n_iter, best_it, best_time, n_g_evals, best_ng

                    if alpha > 0:
                        succ[i_dir] = succ[i_dir] + 1

                        if all_ones_for_ds >= 1:
                            all_ones_for_ds = 0

                        D[:, i_dir] = d
                        y = x_trial
                        fy = f_trial
                        alpha_tilde[i_dir] = alpha

                        int_moved = True

                        if fy < best_f - self.tolerance_for_best:
                            best_x = y
                            best_f = fy
                            best_nf = n_f_evals
                            best_ng = n_g_evals
                            best_it = n_iter
                            best_time = time.time() - start_time

                    else:
                        alpha_tilde[i_dir] = max(1, np.floor(alpha_tilde[i_dir] / 2))

                else:
                    succ[i_dir] = succ[i_dir] + 1

                    if all_ones_for_ds >= 1:
                        all_ones_for_ds = 0

                    y = x_trial
                    fy = f_trial
                    alpha_tilde[i_dir] = alpha

                    int_moved = True

                    if fy < best_f - self.tolerance_for_best:
                        best_x = y
                        best_f = fy
                        best_nf = n_f_evals
                        best_ng = n_g_evals
                        best_it = n_iter
                        best_time = time.time() - start_time

                if self.verbose:
                    print(print_format % ('(d)', n_iter, n_f_evals, fy, max(alpha_tilde), i_dir + 1, np.shape(D)[1]))

                if all_ones_for_ds >= 1:
                    break

            if np.linalg.norm(y - x) <= 1e-14 and max(alpha_tilde) == 1 and old_max_alpha == 1:

                xi = max(xi / 2, self.min_xi_for_discrete_dir)

                all_ones_for_ds = all_ones_for_ds + 1

                iexit = 0

                while iexit == 0:

                    D, succ, alpha_tilde, iexit = self.generate_dirs(problem.ncont, problem.nint, D, succ, alpha_tilde, eta_for_dir, sequencer)

                    if iexit == 0:
                        eta_for_dir = eta_for_dir + 0.5

                    if eta_for_dir >= 0.5 * (np.linalg.norm(problem.ub - problem.lb) / 2):
                        return best_x, best_f, "Useful dicrete directions already visited", n_f_evals, best_nf, n_iter, best_it, best_time, n_g_evals, best_ng

            if np.linalg.norm(y - x) <= 1e-14 and np.shape(D)[1] >= self.max_num_discret_dirs:
                return best_x, best_f, "Max discrete directions reached", n_f_evals, best_nf, n_iter, best_it, best_time, n_g_evals, best_ng

            if n_f_evals >= self.max_fun:
                return best_x, best_f, "Max function evaluations reached", n_f_evals, best_nf, n_iter, best_it, best_time, n_g_evals, best_ng

            x = y
            f = fy
            old_max_alpha = np.max(alpha_tilde)