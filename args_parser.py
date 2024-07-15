import sys
import argparse


def get_args():

    parser = argparse.ArgumentParser(prog='run_all', description='Run DFO codes')

    parser.add_argument('--seeds', help='Seeds', type=int, nargs='+')
    parser.add_argument('--verbose', help='Verbose', action='store_true', default=False)
    parser.add_argument('--save_logs', help='Save logs in files', action='store_true', default=False)

    parser.add_argument('--max_time', help='Maximum number of elapsed seconds', type=float)
    parser.add_argument('--max_fun', help='Maximum number of function evaluations', type=int)
    parser.add_argument('--max_it', help='Maximum number of iterations', type=int)
    parser.add_argument('--max_num_discrete_dirs', help='Maximum number of directions for integer variables', type=int, default=300)

    parser.add_argument('--type_gradient_related_direction', help='Type of gradient-related direction for continuous variables', type=str, nargs='+', default=['lbfgs'], choices=['lbfgs', 'wolfe', 'pj'])
    parser.add_argument('--max_continuous_iter', help='Maximum number of consecutive iterations for continuous variables', type=int, nargs='+', default=[1])
    
    parser.add_argument('--tolerance_for_best', help='Tolerance for new best solution', type=float, default=1e-7)
    
    parser.add_argument('--tolerance_for_continuous_dir', help='Tolerance for gradient-related direction', type=float, default=1e-6)
    parser.add_argument('--armijo_alpha_0', help='Starting value for alpha in Armijo-type line search', type=float, default=1)
    parser.add_argument('--armijo_gamma', help='Gamma parameter for Armijo-type line search', type=float, default=1e-4)
    parser.add_argument('--armijo_delta', help='Delta parameter for Armijo-type line search', type=float, default=0.5)
    parser.add_argument('--armijo_min_alpha', help='Minimum value for alpha in Armijo-type line search', type=float, default=1e-7)
    
    parser.add_argument('--eta_for_discrete_dir', help='Eta parameter for discerete directions', type=float, default=1.5)
    parser.add_argument('--xi_for_discrete_dir', help='Xi parameter for discerete directions', type=float, default=1.0)
    parser.add_argument('--min_xi_for_discrete_dir', help='Minimum value for xi parameter', type=float, default=1e-7)

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
    assert args.armijo_alpha_0 > 0
    assert 0 < args.armijo_gamma < 1
    assert 0 < args.armijo_delta < 1
    assert 0 < args.armijo_min_alpha < args.armijo_alpha_0

    assert args.eta_for_discrete_dir > 0
    assert args.xi_for_discrete_dir > 0
    assert 0 < args.min_xi_for_discrete_dir < args.xi_for_discrete_dir