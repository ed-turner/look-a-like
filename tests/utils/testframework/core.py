import unittest
import numpy as np


class CoreTestFrameWork(unittest.TestCase):

    def assert_all_close_numpy(self, arr1, arr2):
        """

        :param arr1:
        :param arr2:
        :return:
        """

        assert np.allclose(arr1, arr2)

