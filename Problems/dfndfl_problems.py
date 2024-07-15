import numpy as np
import tensorflow as tf

from Problems.problem_class import problem


#**************************************************
# prob. n.15 described in paper:
# J. Müller, C.A. Shoemaker, R. Piché
# SO-I: a surrogate model algorithm for expensive nonlinear
# integer programming problems including global optimization applications
# Journal of Global Optimization, 59(4):865-889 (2014)
#**************************************************
class problem115(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='prob115',
            n=n,
            nint=nint,
            lb=-10.0 * np.ones(n),
            ub=30.0 * np.ones(n)
        )

        self.objectives = tf.reduce_sum([self._z[i] ** 2 - tf.cos(2 * np.pi * self._z[i]) for i in range(self.n)])


#**************************************************
# prob. n.6 described in paper:
# J. Müller
# MISO: Mixed-Integer Surrogate Optimization Framework
# Optimization and Engineering, 17(1):177-203 (2016)
#**************************************************
class problem206(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='prob206',
            n=n,
            nint=nint,
            lb=-15.0 * np.ones(n),
            ub=30.0 * np.ones(n)
        )

        self.objectives = -20*tf.exp(-0.2*tf.sqrt(tf.reduce_sum([self._z[i]**2 for i in range(self.n)])/self.n)) - tf.exp(tf.reduce_sum([tf.cos(2*np.pi*self._z[i]) for i in range(self.n)])/self.n)


#**************************************************
# prob. n.8 described in paper:
# J. Müller
# MISO: Mixed-Integer Surrogate Optimization Framework
# Optimization and Engineering, 17(1):177-203 (2016)
#**************************************************
class problem208(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='prob208',
            n=n,
            nint=nint,
            lb=-15.0 * np.ones(n),
            ub=30.0 * np.ones(n)
        )

        self.objectives = (self._z[0] - 1) ** 2 + tf.reduce_sum([(i + 1) * (2 * self._z[i] ** 2 - self._z[i - 1]) ** 2 for i in range(1, self.n)])
