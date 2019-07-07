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

        def decorated(*args, **kwargs):
            for arg in args:
                assert isinstance(arg, assert_type)

            for val in kwargs.values():
                assert isinstance(val, assert_type)

            return funct(*args, **kwargs)

        decorated.__name__ = funct.__name__

        return decorated


class AssertArgumentNDArray(AssertArgumentTypeBase):

    assert_type = np.ndarray

