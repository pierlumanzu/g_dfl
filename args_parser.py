import sys
import argparse


def get_args():

    parser = argparse.ArgumentParser(prog='run_all', description='Run DFO codes')

    parser.add_argument('--algs', help='Algorithms', type=str, nargs='+', choices=['G-DFL', 'MISQP'])

    parser.add_argument('--seeds', help='G-DFL parameter -- Seeds', type=int, nargs='+')
    parser.add_argument('--verbose', help='Verbose', action='store_true', default=False)
    parser.add_argument('--save_logs', help='Save logs in files', action='store_true', default=False)

    parser.add_argument('--max_time', help='Maximum number of elapsed seconds', type=float)
    parser.add_argument('--max_fun', help='G-DFL parameter -- Maximum number of function evaluations', type=int)
    parser.add_argument('--max_it', help='G-DFL parameter -- Maximum number of iterations', type=int)
    parser.add_argument('--max_num_discrete_dirs', help='G-DFL parameter -- Maximum number of directions for integer variables', type=int, default=300)

    parser.add_argument('--type_gradient_related_direction', help='G-DFL parameter -- Type of gradient-related direction for continuous variables', type=str, nargs='+', default=['lbfgs'], choices=['lbfgs', 'wolfe', 'pj'])
    parser.add_argument('--max_continuous_iter', help='G-DFL parameter -- Maximum number of consecutive iterations for continuous variables', type=int, nargs='+', default=[1])
    
    parser.add_argument('--tolerance_for_best', help='Tolerance for new best solution', type=float, default=1e-7)
    
    parser.add_argument('--tolerance_for_continuous_dir', help='G-DFL parameter -- Tolerance for gradient-related direction', type=float, default=1e-6)
    parser.add_argument('--armijo_gamma', help='G-DFL parameter -- Gamma parameter for Armijo-type line search', type=float, default=1e-4)
    parser.add_argument('--armijo_delta', help='G-DFL parameter -- Delta parameter for Armijo-type line search', type=float, default=0.5)
    parser.add_argument('--armijo_min_alpha', help='G-DFL parameter -- Minimum value for alpha in Armijo-type line search', type=float, default=1e-7)
    
    parser.add_argument('--eta_for_discrete_dir', help='G-DFL parameter -- Eta parameter for discerete directions', type=float, default=1.5)
    parser.add_argument('--xi_for_discrete_dir', help='G-DFL parameter -- Xi parameter for discerete directions', type=float, default=1.0)
    parser.add_argument('--min_xi_for_discrete_dir', help='G-DFL parameter -- Minimum value for xi parameter', type=float, default=1e-7)

    parser.add_argument('--initial_delta_c', help='MISQP parameter -- Initial delta for continuous variables', type=float, default=10)
    parser.add_argument('--initial_delta_i', help='MISQP parameter -- Initial delta for discrete variables', type=float, default=10)
    parser.add_argument('--epsilon', help='MISQP parameter -- Tolerance for directions', type=float, default=1e-6)
    parser.add_argument('--M', help='MISQP parameter -- Memory of successfull iterations', type=int, default=10)

    return parser.parse_args(sys.argv[1:])


def check_args(args):
    
    assert args.seeds is not None
    for s in args.seeds:
        assert s >= 0

    if args.max_time is not None:
        assert args.max_time > 0

    if args.max_fun is not None:
        assert args.max_fun > 0

    if args.max_it is not None:
        assert args.max_it > 0

    assert args.max_num_discrete_dirs > 0

    for mci in args.max_continuous_iter:
        assert mci > 0 

    assert args.tolerance_for_best >= 0

    assert args.tolerance_for_continuous_dir >= 0
    assert 0 < args.armijo_gamma < 1
    assert 0 < args.armijo_delta < 1
    assert 0 < args.armijo_min_alpha < 1

    assert args.eta_for_discrete_dir > 0
    assert args.xi_for_discrete_dir > 0
    assert 0 < args.min_xi_for_discrete_dir < args.xi_for_discrete_dir

    assert args.initial_delta_c > 0
    assert args.initial_delta_i > 0
    assert args.epsilon > 0
    assert args.M >= 0