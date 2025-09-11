import numpy as np
import time
from gurobipy import Model, GRB, GurobiError


class MISQP:

    def __init__(self,
                 verbose,
                 max_time, 
                 tolerance_for_best,
                 initial_delta_c, initial_delta_i,
                 epsilon, M):
        
        self.verbose = verbose

        self.max_time = max_time

        self.tolerance_for_best = tolerance_for_best

        self.initial_delta_c = initial_delta_c
        self.initial_delta_i = initial_delta_i

        self.epsilon = epsilon
        self.M = M

    def solve(self, prob, start_time):

        if prob.nint <= 1:
            raise AssertionError("ERROR: number of integer variables must be > 1.")

        ub_int = np.copy(prob.ub)
        lb_int = np.copy(prob.lb)
        ub_int[prob.ncont:] = np.round(ub_int[prob.ncont:])
        lb_int[prob.ncont:] = np.round(lb_int[prob.ncont:])

        if np.sum(np.abs(prob.ub[prob.ncont:] - ub_int[prob.ncont:])) != 0 or np.sum(np.abs(prob.lb[prob.ncont:] - lb_int[prob.ncont:])) != 0:
            raise AssertionError("ERROR: upper and/or lower bound on some variable is NOT integer.")

        if np.min((prob.x_initial >= prob.lb) & (prob.x_initial <= prob.ub)) == 0:
            raise RuntimeError("ERROR: Initial point does not satisfy the bound constraints.")

        n_f_evals = 0
        n_g_evals = 0
        n_iter = 0

        x = np.copy(prob.x_initial)
        x[prob.ncont:] = np.round(x[prob.ncont:])
        f = prob.feval(x)
        n_f_evals += 1
        gr = prob.geval(x)
        n_g_evals += 1

        best_f = np.inf
        best_x = np.copy(x)
        best_nf = n_f_evals
        best_ng = n_g_evals
        best_it = n_iter
        best_time = time.time() - start_time

        if self.verbose:
            print_format = '%+13.8e|   %5d |        %5d | %+13.8e | %+13.8e | %+13.8e | %+13.8e |   '
            print('       time    |  n_iter |    n_f_evals |          f      |        best_f   |      delta_c    |      delta_i    |')

        x_restart = None
        f_restart = None

        delta_c = self.initial_delta_c
        delta_i = self.initial_delta_i
        x_1 = None
        gr_1 = None
        vw_l = None
        vw_u = None
        C = np.eye(prob.n)

        K = f * np.ones(self.M)

        gr, f_bn, x_bn, n_f_evals = self.approximate_integer_gradient(prob, x, f, gr, n_f_evals)

        if f_bn < best_f - self.tolerance_for_best:
            best_f = f_bn
            best_x = np.copy(x_bn)
            best_nf = n_f_evals
            best_ng = n_g_evals
            best_it = n_iter
            best_time = time.time() - start_time

        while True:

            n_iter += 1

            if x_1 is not None and gr_1 is not None and vw_l is not None and vw_u is not None:
                s = x - x_1
                y = gr - gr_1
                
                ys = np.dot(y, s)
                Bs = np.dot(C, s)
                sBs = np.dot(s, Bs)
                
                theta = 0.8 * sBs / (sBs - ys) if ys < 0.2 * sBs else 1                    
                y = theta * y + (1 - theta) * Bs

                C = C - np.outer(Bs, Bs) / sBs + np.outer(y, y) / (np.dot(y, s) if theta != 1 else ys)
            else:
                assert x_1 is None and gr_1 is None and vw_l is None and vw_u is None

            dir, Phi_dir = self.get_direction(prob, x, gr, C, delta_c, delta_i, start_time)

            if Phi_dir < -self.epsilon and time.time() - start_time <= self.max_time:
                x_new_tmp = x + dir
                f_new_tmp = prob.feval(x_new_tmp)
                n_f_evals += 1

                ratio = (np.max(K) - f_new_tmp) / -Phi_dir

                if ratio <= 0.75:
                    dir_hat = self.SOC_step(prob, x, dir, gr, C, delta_c, start_time)
                    
                    x_new = x_new_tmp + dir_hat
                    f_new = prob.feval(x_new)
                    n_f_evals += 1

                    if f_new < f_new_tmp:
                        ratio = (np.max(K) - f_new) / -Phi_dir
                        dir += dir_hat
                    else:
                        x_new = np.copy(x_new_tmp)
                        f_new = f_new_tmp

                else:
                    x_new = np.copy(x_new_tmp)
                    f_new = f_new_tmp

                if ratio < 0.25:
                    delta_c = min(np.linalg.norm(dir, ord=np.inf) / 2, delta_c)
                    delta_i = np.floor(np.linalg.norm(dir[prob.ncont:], ord=np.inf) / 2)
                elif ratio > 0.75:
                    delta_c = max(np.linalg.norm(dir, ord=np.inf) * 2, delta_c)
                    delta_i = max(np.linalg.norm(dir[prob.ncont:], ord=np.inf) * 2, delta_i, 1)

                if ratio <= 0:
                    x_1 = None
                    gr_1 = None
                    vw_l = None
                    vw_u = None
                    
                    if self.verbose:
                        print(print_format % (time.time() - start_time, n_iter, n_f_evals, f, best_f, delta_c, delta_i))
                
                else:
                    x_1 = np.copy(x)
                    gr_1 = np.copy(gr)

                    x = np.copy(x_new)
                    f = f_new

                    K = np.roll(K, 1)
                    K[0] = f
                    
                    gr = prob.geval(x)
                    n_g_evals += 1

                    gr, f_bn, x_bn, n_f_evals = self.approximate_integer_gradient(prob, x, f, gr, n_f_evals)

                    if f_bn < best_f - self.tolerance_for_best:
                        best_f = f_bn
                        best_x = np.copy(x_bn)
                        best_nf = n_f_evals
                        best_ng = n_g_evals
                        best_it = n_iter
                        best_time = time.time() - start_time

                    vw_l, vw_u = self.get_lagrange_multipliers(prob, x, gr, start_time)

                    if self.verbose:
                        print(print_format % (time.time() - start_time, n_iter, n_f_evals, f, best_f, delta_c, delta_i))
   
            else:
                
                C = C / np.linalg.norm(C, ord=np.inf) * np.linalg.norm(gr, ord=np.inf) / prob.n

                if f > best_f and not np.isinf(f) and time.time() - start_time <= self.max_time:
                    x_1 = None
                    gr_1 = None
                    vw_l = None
                    vw_u = None

                    x = np.copy(best_x)
                    f = best_f

                    gr = prob.geval(x)
                    n_g_evals += 1

                    gr, f_bn, x_bn, n_f_evals = self.approximate_integer_gradient(prob, x, f, gr, n_f_evals)

                    if f_bn < best_f - self.tolerance_for_best:
                        best_f = f_bn
                        best_x = np.copy(x_bn)
                        best_nf = n_f_evals
                        best_ng = n_g_evals
                        best_it = n_iter
                        best_time = time.time() - start_time

                    if self.verbose:
                        print(print_format % (time.time() - start_time, n_iter, n_f_evals, f, best_f, delta_c, delta_i))

                else:

                    if f <= best_f:
                        
                        if f < best_f - self.tolerance_for_best:
                            best_f = f
                            best_x = np.copy(x)
                            best_nf = n_f_evals
                            best_ng = n_g_evals
                            best_it = n_iter
                            best_time = time.time() - start_time
                    
                    else:
                        print("Warning: Problem might be infeasible or there is no other available time")

                    if x_restart is not None and f_restart is not None:
                        if abs(f - f_restart) < self.tolerance_for_best:
                            return best_x, best_f, 'Restart strategy failed', n_f_evals, best_nf, n_iter, best_it, best_time, n_g_evals, best_ng
                    else:
                        assert x_restart is None and f_restart is None
                    
                    print('Restart Strategy')

                    x_restart = np.copy(x)
                    f_restart = f

                    delta_c = self.initial_delta_c
                    delta_i = self.initial_delta_i
                    x_1 = None
                    gr_1 = None
                    vw_l = None
                    vw_u = None
                    C = np.eye(prob.n)

                    dir, Phi_dir = self.get_direction(prob, x, gr, C, delta_c, delta_i, start_time, relaxed=True)

                    if Phi_dir >= -self.epsilon:
                        return best_x, best_f, 'Max time reached' if time.time() - start_time > self.max_time else 'Solved', n_f_evals, best_nf, n_iter, best_it, best_time, n_g_evals, best_ng

                    x += dir
                    f = prob.feval(x)
                    n_f_evals += 1
                    
                    gr = prob.geval(x)
                    n_g_evals += 1

                    K = f * np.ones(self.M)

                    gr, f_bn, x_bn, n_f_evals = self.approximate_integer_gradient(prob, x, f, gr, n_f_evals)

                    if f_bn < best_f - self.tolerance_for_best:
                        best_f = f_bn
                        best_x = np.copy(x_bn)
                        best_nf = n_f_evals
                        best_ng = n_g_evals
                        best_it = n_iter
                        best_time = time.time() - start_time

                    if self.verbose:
                        print(print_format % (time.time() - start_time, n_iter, n_f_evals, f, best_f, delta_c, delta_i))    

    @staticmethod
    def approximate_integer_gradient(prob, x, f, gr, n_f_evals):
        
        f_bn = np.inf
        x_bn = np.copy(x)
        
        for i in range(prob.ncont, prob.n):
            z_plus = np.copy(x)
            z_plus[i] += 1
            z_minus = np.copy(x)
            z_minus[i] -= 1

            if prob.lb[i] < x[i] < prob.ub[i]:
                f_plus = prob.feval(z_plus)
                f_minus = prob.feval(z_minus)
                n_f_evals += 2

                for z_pm, f_pm in zip([z_plus, z_minus], [f_plus, f_minus]):
                    if f_pm < f_bn:
                        f_bn = f_pm
                        x_bn = np.copy(z_pm)

                gr[i] = (f_plus - f_minus) / 2
            
            elif x[i] == prob.lb[i]:
                f_plus = prob.feval(z_plus)
                n_f_evals += 1

                if f_plus < f_bn:
                    f_bn = f_plus
                    x_bn = np.copy(z_plus)

                gr[i] = f_plus - f
            
            else:  # x[i] == prob.ub[i]
                f_minus = prob.feval(z_minus)
                n_f_evals += 1

                if f_minus < f_bn:
                    f_bn = f_minus
                    x_bn = np.copy(z_minus)

                gr[i] = f - f_minus

        return gr, f_bn, x_bn, n_f_evals
    
    def get_direction(self, prob, x, gr, C, delta_c, delta_i, start_time, relaxed=False):
        
        if (np.isinf(gr).any() or np.isnan(gr).any()) or (np.isinf(C).any() or np.isnan(C).any()):
            return np.zeros(prob.n), 0
        
        try:
            model = Model("MISQP Problem to get direction")

            model.setParam("OutputFlag", False)
            model.setParam("TimeLimit", max(0, self.max_time - (time.time() - start_time)))

            d_c = model.addMVar(prob.ncont, 
                                lb=np.maximum(prob.lb[:prob.ncont] - x[:prob.ncont], -delta_c * np.ones(prob.ncont)), 
                                ub=np.minimum(prob.ub[:prob.ncont] - x[:prob.ncont], delta_c * np.ones(prob.ncont)), 
                                name="d_c")
            d_i = model.addMVar(prob.nint, vtype=(GRB.INTEGER if not relaxed else GRB.CONTINUOUS), 
                                lb=np.maximum(prob.lb[prob.ncont:] - x[prob.ncont:], -delta_i * np.ones(prob.nint)), 
                                ub=np.minimum(prob.ub[prob.ncont:] - x[prob.ncont:], delta_i * np.ones(prob.nint)), 
                                name="d_i")

            model.setObjective(gr[:prob.ncont] @ d_c + gr[prob.ncont:] @ d_i + 0.5 *
                               (d_c @ C[:prob.ncont, :prob.ncont] @ d_c + d_c @ C[:prob.ncont, prob.ncont:] @ d_i + 
                                d_i @ C[prob.ncont:, :prob.ncont] @ d_c + d_i @ C[prob.ncont:, prob.ncont:] @ d_i))
            
            model.update()
            model.optimize()

            if model.Status == GRB.OPTIMAL:
                dir = np.array([s.x for s in model.getVars()])
                if not relaxed:
                    return dir, model.getObjective().getValue()
                else:
                    dir[prob.ncont:] = np.round(dir[prob.ncont:])
                    return dir, np.dot(gr, dir) + 0.5 * np.dot(dir, np.dot(C, dir))
            else:
                return np.zeros(prob.n), 0
            
        except GurobiError:
            return np.zeros(prob.n), 0

    def SOC_step(self, prob, x, dir, gr, C, delta_c, start_time):
        
        if (np.isinf(gr).any() or np.isnan(gr).any()) or (np.isinf(C).any() or np.isnan(C).any()):
            return np.zeros(prob.n)
        
        try:
            model = Model("SOC Step Problem")

            model.setParam("OutputFlag", False)
            model.setParam("TimeLimit", max(0, self.max_time - (time.time() - start_time)))

            d_c = model.addMVar(prob.ncont, 
                                lb=np.maximum(prob.lb[:prob.ncont] - (x[:prob.ncont] + dir[:prob.ncont]), -delta_c * np.ones(prob.ncont) - dir[:prob.ncont]), 
                                ub=np.minimum(prob.ub[:prob.ncont] - (x[:prob.ncont] + dir[:prob.ncont]), delta_c * np.ones(prob.ncont) - dir[:prob.ncont]), 
                                name="d_c")
            d_concat = model.addMVar(prob.ncont, lb=-np.inf, ub=np.inf, name="d_concat")
            
            model.setObjective(gr[:prob.ncont] @ d_concat + 0.5 * 
                               (d_concat @ C[:prob.ncont, :prob.ncont] @ d_concat + d_concat @ C[:prob.ncont, prob.ncont:] @ dir[prob.ncont:] + 
                                dir[prob.ncont:] @ C[prob.ncont:, :prob.ncont] @ d_concat))
            
            model.addConstr(d_concat == d_c + dir[:prob.ncont], name='d_concat constraint')

            model.update()
            model.optimize()

            if model.Status == GRB.OPTIMAL:
                return np.concatenate((np.array([s.x for s in model.getVars()][:prob.ncont]), np.zeros(prob.nint)))
            else:
                return np.zeros(prob.n)
            
        except GurobiError:
            return np.zeros(prob.n)
        
    def get_lagrange_multipliers(self, prob, x, gr, start_time):
        
        if (np.isinf(gr).any() or np.isnan(gr).any()):
            return np.zeros(prob.n), np.zeros(prob.n)
        
        try:
            model = Model("Lagrange Multipliers Problem")

            model.setParam("OutputFlag", False)
            model.setParam("TimeLimit", max(0, self.max_time - (time.time() - start_time)))

            vw_l = model.addMVar(prob.n, 
                                 lb=np.zeros(prob.n), ub=np.array([np.inf if x[i] == prob.lb[i] else 0 for i in range(prob.n)]), 
                                 name="vw_l")
            vw_u = model.addMVar(prob.n, 
                                 lb=np.zeros(prob.n), ub=np.array([np.inf if x[i] == prob.ub[i] else 0 for i in range(prob.n)]), 
                                 name="vw_u")
            vw_concat = model.addMVar(prob.n, lb=-np.inf, ub=np.inf, name="vw_concat")
            
            model.setObjective(2 * (gr @ vw_concat) + vw_concat @ vw_concat)
            
            model.addConstr(vw_concat == vw_u - vw_l, name='vw_concat constraint')

            model.update()
            model.optimize()

            if model.Status == GRB.OPTIMAL:
                sol = np.array([s.x for s in model.getVars()])
                return sol[:prob.n], sol[prob.n:2*prob.n]
            else:
                return np.zeros(prob.n), np.zeros(prob.n)
            
        except GurobiError:
            return np.zeros(prob.n), np.zeros(prob.n)
            
            

            
            

        


