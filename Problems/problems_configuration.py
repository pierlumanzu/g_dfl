from Problems.dfndfl_problems import problem115, problem206, problem208
from Problems.cute_problems import bdexp, biggsb1, chenhark, cvxbqp1, explin, explin2, expquad, hs110, mccormck, ncvxbqp1, ncvxbqp2, ncvxbqp3, nonscomp, pentdi, probpenl, qudlin, sineali


PROBLEMS = [(p, n, nint)
            for n, nint in [(100, 2), (200, 4), (500, 10), 
                            (1000, 20), (2000, 40), (5000, 100)]
            for p in [problem115, problem206, problem208, bdexp, biggsb1, 
                      chenhark, cvxbqp1, explin, explin2, expquad, 
                      hs110, mccormck, ncvxbqp1, ncvxbqp2, ncvxbqp3, 
                      nonscomp, pentdi, probpenl, qudlin, sineali]]
