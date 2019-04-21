from abc import ABCMeta, abstractmethod

import numpy as np

from numba import jit


class KNNBase(metaclass=ABCMeta):

    def __init__(self, k):
        self.k = k

    @abstractmethod
    def calc_dist(self, mat1, mat2):
        pass

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

    def knn_match(self, mat1, mat2):
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


class KNNPowerMatcher(KNNBase):

    def __init__(self, k, p):
        """

        :param k:
        :param p:
        """
        super().__init__(k)
        self.p = p
        pass

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


class KNNCosineMatcher(KNNBase):

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
