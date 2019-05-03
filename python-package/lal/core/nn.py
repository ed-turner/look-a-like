from abc import ABCMeta, abstractmethod, ABC

import numpy as np
from scipy.optimize import linear_sum_assignment

from numba import jit


class DistanceBase(metaclass=ABCMeta):

    @abstractmethod
    def calc_dist(self, mat1, mat2):
        pass


class PowerDistanceBase(DistanceBase):

    def __init__(self, p):
        self.p = p

    @staticmethod
    @jit(nopython=True)
    def _calc_dist(mat1, mat2, p):
        """

        :param mat1:
        :param mat2:
        :return:
        """
        abs_dist = np.abs(mat1.reshape(mat1.shape + (1,)) - mat2.reshape(mat2.shape + (1,)).T) ** p

        return np.sum(abs_dist, axis=1) ** (1.0 / p)

    def calc_dist(self, mat1, mat2):
        return self._calc_dist(mat1, mat2, self.p)


class CosineDistanceBase(DistanceBase):

    @staticmethod
    @jit(nopython=True)
    def _calc_dist(mat1, mat2):
        """

        :param mat1:
        :param mat2:
        :return:
        """

        n1 = mat1.shape[0]
        n2 = mat2.shape[0]

        res = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                norm_1 = np.linalg.norm(mat1[i, :])
                norm_2 = np.linalg.norm(mat2[j, :])

                bot = norm_1 * norm_2
                top = np.dot(mat1[i, :], mat2[j, :])

                res[i, j] = 1.0 - (top / bot)
                res[j, i] = res[i, j]

        return res

    def calc_dist(self, mat1, mat2):
        return self._calc_dist(mat1, mat2)


class KNNBase(DistanceBase, ABC):

    def __init__(self, k):
        self.k = k

    @staticmethod
    @jit(nopython=True)
    def _get_k_neighbors(dist, k):
        """

        :param dist:
        :return:
        """

        n = dist.shape[0]

        res_indices = np.zeros((1, k))

        for i in range(n):
            k_match = np.argsort(dist[i, :])[:k]
            res_indices = np.vstack((res_indices, k_match.reshape(1, -1)))

        return res_indices[1:, :]

    def _knn_match_batch(self, mat1_batch, mat2_batch, k):
        """

        :param mat1_batch:
        :param mat2_batch:
        :return:
        """

        dist = self.calc_dist(mat1_batch, mat2_batch)

        return self._get_k_neighbors(dist, k)

    def match(self, mat1, mat2):
        """

        :param mat1:
        :param mat2:
        :return:
        """

        n1 = mat1.shape[0]
        n2 = mat2.shape[0]

        res_lst = np.zeros((1, self.k)).astype(np.int64)

        for i in range(0, n1, 100):
            tmp_i_indices = np.arange(i, min(i + 100, n1))

            mat1_batch = np.ascontiguousarray(mat1[tmp_i_indices, :])

            unique_i_batch_indices = np.zeros((1, self.k))

            for j in range(0, n2, 100):
                tmp_j_indices = np.arange(j, min(j + 100, n2))

                mat2_batch = np.ascontiguousarray(mat1[tmp_j_indices, :])

                indices = self._knn_match_batch(mat1_batch, mat2_batch, self.k)

                for k in range(indices.shape[0]):
                    unique_i_batch_indices = np.vstack((unique_i_batch_indices, indices))

            unique_mat2_batch_lst = np.unique(
                np.ascontiguousarray(
                    unique_i_batch_indices[1:, :].reshape(-1, )
                ).astype(np.int64)
            )

            mat2_batch = mat2[unique_mat2_batch_lst, :]

            tmp_indices = self._knn_match_batch(mat1_batch, mat2_batch, self.k)

            for k in range(tmp_indices.shape[0]):
                k_indices = np.ascontiguousarray(tmp_indices[k, :].reshape(-1, )).astype(np.int64)

                res_lst = np.vstack((res_lst,
                                     np.ascontiguousarray(unique_mat2_batch_lst[k_indices].reshape(-1, self.k))))

        return res_lst[1:, :]


class KNNPowerMatcher(PowerDistanceBase, KNNBase):

    def __init__(self, k, p):
        PowerDistanceBase.__init__(self, p)
        KNNBase.__init__(self, k)


class KNNCosineMatcher(CosineDistanceBase, KNNBase):

    def __init__(self, k):
        KNNBase.__init__(self, k)


class NNLinearSumBase(DistanceBase):

    def __init__(self):
        super().__init__()

    def match(self, mat1, mat2):
        """
        Get all samples in mat2 to match to mat1
        :param mat1:
        :param mat2:
        :return:
        """

        dist = self.calc_dist(mat1, mat2)

        match_lst = []

        mat1_indices = np.arange(mat1.shape[0])

        while True:
            x_indices, y_indices = linear_sum_assignment(dist[mat1_indices, :])

            matches = np.column_stack((mat1_indices[x_indices].reshape(-1, 1),
                                       y_indices.reshape(-1, 1)))

            match_lst.append(matches)

            C = np.searchsorted(mat1_indices, matches[:, 0])
            D = np.delete(np.arange(np.alen(mat1_indices)), C)

            mat1_indices = mat1_indices[D]

            if mat1_indices.shape[0] == 0:
                break

        matches = np.vstack(match_lst)

        return matches[np.argsort(matches[:, 0]), :]


class NNLinearSumPowerMatcher(PowerDistanceBase, NNLinearSumBase):

    def __init__(self, p):
        PowerDistanceBase.__init__(self, p)


class NNLinearSumCosineMatcher(CosineDistanceBase, NNLinearSumBase):
    pass
