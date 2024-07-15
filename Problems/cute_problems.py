# Python adaptation of the AMPL model by Hande Y. Benson
# Copyright (C) 2001 Princeton University
# All Rights Reserved

import numpy as np
import tensorflow as tf

from Problems.problem_class import problem


class bdexp(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='bdexp',
            n=n,
            nint=nint,
            lb=-2 * np.ones(n),
            ub=2 * np.ones(n)
        )

        self.objectives = tf.reduce_sum([(self._z[i]+self._z[i+1])*tf.exp((self._z[i]+self._z[i+1])*(-self._z[i+2])) for i in range(self.n - 2)])


class biggsb1(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='biggsb1',
            n=n,
            nint=nint,
            lb=0 * np.ones(n),
            ub=0.9 * np.ones(n)
        )
        self.ub[self.ncont:] = 1

        self.objectives = (self._z[0] - 1)**2 + tf.reduce_sum([(self._z[i + 1] - self._z[i])**2 for i in range(self.n - 1)]) + (1 - self._z[self.n - 1])**2


class chenhark(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='chenhark',
            n=n,
            nint=nint,
            lb=0.0 * np.ones(n),
            ub=100.0 * np.ones(n),
            set_x_initial=False
        )
        self.x_initial = 0.5 * np.ones(n)

        self.x_p = np.array([0.0 if (i - 1 <= 0 or i - 1 > self.n / 2) else 1.0 for i in range(self.n + 4)])

        self.objectives = tf.reduce_sum([0.5*(self._z[i+1] + self._z[i-1] - 2*self._z[i])**2 for i in range(1, self.n - 1)]) + 0.5*self._z[0]**2 + 0.5*(2*self._z[0] - self._z[1])**2 + 0.5*(2*self._z[self.n - 1] - self._z[self.n-2])**2 + 0.5*(self._z[self.n - 1])**2 + tf.reduce_sum([self._z[i] * (-6*self.x_p[i+2] + 4*self.x_p[i+3] + 4*self.x_p[i+1] - self.x_p[i+4] - self.x_p[i]) for i in range(self.n//2 + self.n//5)]) + tf.reduce_sum([self._z[i] * (-6*self.x_p[i+2] + 4*self.x_p[i+3] + 4*self.x_p[i+1] - self.x_p[i+4] - self.x_p[i] + 1) for i in range(self.n//2 + self.n//5, self.n)])


class cvxbqp1(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='cvxbqp1',
            n=n,
            nint=nint,
            lb=0.1 * np.ones(n),
            ub=10 * np.ones(n),
            set_x_initial=False
        )
        self.lb[self.ncont:] = 1
        self.x_initial = 0.5 * np.ones(self.n)
        self.x_initial[self.ncont:] = 1

        self.objectives = tf.reduce_sum([0.5 * (i + 1) * (self._z[i] + self._z[(2*(i + 1)-1) % self.n] + self._z[(3*(i + 1)-1) % self.n])**2 for i in range(self.n)])


class explin(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='explin',
            n=n,
            nint=nint,
            lb=0.0 * np.ones(n),
            ub=10.0 * np.ones(n),
            set_x_initial=False
        )
        self.x_initial = np.zeros(self.n)

        self.objectives = tf.reduce_sum([tf.exp(0.1 * self._z[i] * self._z[i + 1]) for i in range(self.ncont)]) + tf.reduce_sum([-10.0 * (i + 1) * self._z[i] for i in range(self.n)])


class explin2(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='explin2',
            n=n,
            nint=nint,
            lb=0.0 * np.ones(n),
            ub=10.0 * np.ones(n),
            set_x_initial=False
        )
        self.x_initial = np.zeros(self.n)

        self.objectives = tf.reduce_sum([tf.exp(0.1 * (i + 1) * self._z[i] * self._z[i + 1] / self.ncont) for i in range(self.ncont)]) + tf.reduce_sum([-10.0 * (i + 1) * self._z[i] for i in range(self.n)])


class expquad(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='expquad',
            n=n,
            nint=nint,
            lb=0.0 * np.ones(n),
            ub=4.0 * np.ones(n),
            set_x_initial=False
        )
        self.x_initial = np.zeros(self.n)

        self.objectives = tf.reduce_sum([(4.0 * self._z[i] * self._z[i] + 2.0 * self._z[self.n - 1] * self._z[self.n - 1] + self._z[i] * self._z[self.n - 1]) for i in range(self.ncont)]) + tf.reduce_sum([tf.exp(0.1 * (i + 1 - self.ncont) * self.nint * self._z[i] * self._z[i+1]) for i in range(self.ncont, self.n - 1)]) + tf.reduce_sum([(-10.0 * (i + 1) * self._z[i]) for i in range(self.n)])


class hs110(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='hs110',
            n=n,
            nint=nint,
            lb=2.001 * np.ones(n),
            ub=9.999 * np.ones(n),
            set_x_initial=False
        )
        self.lb[self.ncont:] = 3
        self.ub[self.ncont:] = 9
        self.x_initial = 9 * np.ones(self.n)

        self.objectives = tf.reduce_sum([(tf.math.log(self._z[i]-2)**2 + tf.math.log(10-self._z[i])**2) for i in range(self.n)]) - (tf.reduce_prod([self._z[i] for i in range(self.ncont, self.n)]))**0.2


class mccormck(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='mccormck',
            n=n,
            nint=nint,
            lb=-1.5 * np.ones(n),
            ub=3.0 * np.ones(n)
        )
        self.lb[self.ncont:] = -1

        self.objectives = tf.reduce_sum([(-1.5 * self._z[i] + 2.5 * self._z[i+1] + 1.0 + (self._z[i] - self._z[i+1])**2 + tf.sin(self._z[i] + self._z[i+1])) for i in range(self.n - 1)])


class ncvxbqp1(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='ncvxbqp1',
            n=n,
            nint=nint,
            lb=0.1 * np.ones(n),
            ub=10 * np.ones(n),
            set_x_initial=False
        )
        self.lb[self.ncont:] = 1
        self.x_initial = 0.5 * np.ones(self.n)
        self.x_initial[self.ncont:] = 1

        self.objectives = tf.reduce_sum([0.5*(i + 1)*(self._z[i] + self._z[(2*(i + 1)-1) % self.n] + self._z[(3*(i + 1)-1) % self.n])**2 for i in range(self.n // 4)]) - tf.reduce_sum([0.5*(i + 1)*(self._z[i] + self._z[(2*(i + 1)-1) % self.n] + self._z[(3*(i + 1)-1) % self.n])**2 for i in range(self.n // 4, self.n)])


class ncvxbqp2(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='ncvxbqp2',
            n=n,
            nint=nint,
            lb=0.1 * np.ones(n),
            ub=10 * np.ones(n),
            set_x_initial=False
        )
        self.lb[self.ncont:] = 1
        self.x_initial = 0.5 * np.ones(self.n)
        self.x_initial[self.ncont:] = 1

        self.objectives = tf.reduce_sum([0.5*(i + 1)*(self._z[i] + self._z[(2*(i + 1)-1) % self.n] + self._z[(3*(i + 1)-1) % self.n])**2 for i in range(self.n // 2)]) - tf.reduce_sum([0.5*(i + 1)*(self._z[i] + self._z[(2*(i + 1)-1) % self.n] + self._z[(3*(i + 1)-1) % self.n])**2 for i in range(self.n // 2, self.n)])


class ncvxbqp3(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='ncvxbqp3',
            n=n,
            nint=nint,
            lb=0.1 * np.ones(n),
            ub=10 * np.ones(n),
            set_x_initial=False
        )
        self.lb[self.ncont:] = 1
        self.x_initial = 0.5 * np.ones(self.n)
        self.x_initial[self.ncont:] = 1

        self.objectives = tf.reduce_sum([0.5*(i + 1)*(self._z[i] + self._z[(2*(i + 1)-1) % self.n] + self._z[(3*(i + 1)-1) % self.n])**2 for i in range(3 * self.n // 4)]) - tf.reduce_sum([0.5*(i + 1)*(self._z[i] + self._z[(2*(i + 1)-1) % self.n] + self._z[(3*(i + 1)-1) % self.n])**2 for i in range(3 * self.n // 4, self.n)])


class nonscomp(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='nonscomp',
            n=n,
            nint=nint,
            lb=np.array([1.0 if i % 3 == 0 else -100.0 for i in range(n)]),
            ub=100.0 * np.ones(n),
            set_x_initial=False
        )
        self.x_initial = 3 * np.ones(self.n)

        self.objectives = (self._z[0] - 1)**2 + tf.reduce_sum([4*(self._z[i]-self._z[i-1]**2)**2 for i in range(1, self.n)])


class pentdi(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='pentdi',
            n=n,
            nint=nint,
            lb=0.0 * np.ones(n),
            ub=100 * np.ones(n)
        )

        self.objectives = tf.reduce_sum([6*self._z[i]**2 - 3*self._z[0]+self._z[1]+self._z[self.n//2-2]-3*self._z[self.n//2 - 1] + 4*self._z[self.n//2] for i in range(self.n)]) + tf.reduce_sum([self._z[i] for i in range(self.n//2+2, self.n)]) + tf.reduce_sum([(-4*self._z[i]*self._z[i+1]+self._z[i]*self._z[i+2]) for i in range(self.n - 2)])


class probpenl(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='probpenl',
            n=n,
            nint=nint,
            lb=-5.0 * np.ones(n),
            ub=5.0 * np.ones(n),
            set_x_initial=False
        )
        self.x_initial = 0.5 * np.ones(self.n)

        self.objectives = tf.reduce_sum([(self._z[i]+self._z[i+1])*0.0001*tf.exp(-self._z[i]*self._z[i+1])/self.n for i in range(self.n - 1)]) + 100 * (tf.reduce_sum([self._z[i] for i in range(self.n)]) - 1)**2


class qudlin(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='qudlin',
            n=n,
            nint=nint,
            lb=0.0 * np.ones(n),
            ub=10.0 * np.ones(n)
        )

        self.objectives = tf.reduce_sum([self._z[i] * self._z[i+1] for i in range(self.ncont)]) + tf.reduce_sum([- (i + 1) * 10 * self._z[i] for i in range(self.n)])


class sineali(problem):

    def __init__(self, n, nint):

        problem.__init__(
            self,
            name='sineali',
            n=n,
            nint=nint,
            lb=np.array([-1.5*np.pi if i == 0 else np.sqrt(np.pi)-2*np.pi for i in range(n)]),
            ub=np.array([0.5*np.pi if i == 0 else np.sqrt(np.pi) for i in range(n)]),
            set_x_initial=False
        )
        self.lb[self.ncont:] = np.ceil(self.lb[self.ncont:])
        self.ub[self.ncont:] = np.floor(self.ub[self.ncont:])
        self.x_initial = np.zeros(self.n)

        self.objectives = tf.sin(self._z[0]-1) + tf.reduce_sum([100.0 * tf.sin(self._z[i] - self._z[i-1]**2) for i in range(1, self.n)])
