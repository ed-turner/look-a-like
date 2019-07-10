import numpy as np


class AssertArgumentTypeBase:

    assert_type = None

    def assert_arguments(self, funct):
        """
        Based on the assumption that all the arguments must be the same type, we will decorate the function so that
        we can make sure each argument has the same time.

        :param funct:
        :return:
        """

        assert_type = self.assert_type

        if assert_type is None:
            raise ValueError("The assert_type was not set.")

        # considering we are applying this onto object behaviors, we will ignore the first argument
        def decorated(*args, **kwargs):
            for arg in args[1:]:
                try:
                    assert isinstance(arg, assert_type)
                except AssertionError:
                    raise AssertionError("{} is not a {}".format(arg.__name__, str(assert_type)))

            for val in kwargs.values():
                try:
                    assert isinstance(val, assert_type)
                except AssertionError:
                    raise AssertionError("{} is not a {}".format(val.__name__, str(assert_type)))

            return funct(*args, **kwargs)

        decorated.__name__ = funct.__name__

        return decorated


class AssertArgumentNDArray(AssertArgumentTypeBase):

    assert_type = np.ndarray

