import numpy as np
import tensorflow as tf


class problem:

    def __init__(self, name, n, nint, lb, ub, set_x_initial=True):

        self.name = name

        self.n = n
        self.nint = nint
        self.ncont = n - nint

        self.lb = lb
        self.ub = ub

        if set_x_initial:
            self.x_initial = (self.ub + self.lb) / 2

        self._z = tf.compat.v1.placeholder(dtype=tf.double, shape=[n, ])
        self.__objectives = None
        self.__objectives_gradients = None

    def feval(self, x):
        return self.__objectives.eval({self._z: x})

    def geval(self, x):
        g_num = self.__objectives_gradients.eval({self._z: x})
        g_num[self.ncont:] = 0
        return g_num

    @property
    def objectives(self):
        raise RuntimeError

    @objectives.setter
    def objectives(self, obj):
        self.__objectives = obj
        self.__objectives_gradients = tf.gradients(self.__objectives, self._z)[0]

    def problem_name(self):
        return "{}_{}_{}".format(self.name, self.n, self.nint)
