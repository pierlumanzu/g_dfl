import os
import sys
import time
import numpy as np
import tensorflow as tf

from args_parser import get_args, check_args
from Problems.problems_configuration import PROBLEMS
from G_DFL import G_DFL

args = get_args()
check_args(args)

tf.compat.v1.disable_eager_execution()

g_dfl_instance = G_DFL(args.verbose, args.save_logs, 
                       args.max_time * 60 if args.max_time is not None else np.inf, args.max_fun if args.max_fun is not None else np.inf, args.max_it if args.max_it is not None else np.inf, args.max_num_discrete_dirs,
                       args.tolerance_for_best,
                       args.tolerance_for_continuous_dir, args.armijo_alpha_0, args.armijo_gamma, args.armijo_delta, args.armijo_min_alpha,
                       args.eta_for_discrete_dir, args.xi_for_discrete_dir, args.min_xi_for_discrete_dir)

for problem_item in PROBLEMS:

    session = tf.compat.v1.Session()
    with session.as_default():

        problem = problem_item[0](problem_item[1], problem_item[2])

        problem.feval(problem.x_initial)
        problem.geval(problem.x_initial)

        for seed in args.seeds:

            np.random.seed(seed)

            for tgd in args.type_gradient_related_direction:

                for mci in args.max_continuous_iter:
                    
                    if args.save_logs:
                        if mci == 1:
                            sys.stdout = open(os.path.join("Outputs", "{}-{}.txt".format(tgd.capitalize(), seed)), "a", buffering=1)
                        else:
                            sys.stdout = open(os.path.join("Outputs", "{}-{}-{}.txt".format(tgd.capitalize(), mci, seed)), "a", buffering=1)

                    print("prob: {} | seed: {} | tgd: {} | mci: {}".format(problem.problem_name(), seed, tgd, mci))

                    start_time = time.time()
                    x, _, stop_status, nf, best_nf, n_iter, best_it, best_time, ng, best_ng = g_dfl_instance.run(seed, tgd, mci, problem, start_time)
                    elapsed_time = time.time() - start_time

                    x[problem.ncont:] = np.round(x[problem.ncont:])
                    f = problem.feval(x)

                    print(
                        "f: {} | stop_message: {} | T: {} | T_for_best: {} | n_iter: {} | n_iter_for_best: {} | n_f_evals: {} | n_f_for_best: {} | n_g_evals: {} | n_g_for_best: {}".format(
                            f, stop_status,
                            elapsed_time, best_time,
                            n_iter, best_it,
                            nf, best_nf,
                            ng, best_ng)
                    )
                    if args.verbose:
                        print(' x = ', x)

        tf.compat.v1.reset_default_graph()
        session.close()
