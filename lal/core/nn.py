from abc import ABCMeta, abstractmethod, ABC

import numpy as np
from lap import lapjv

from numba import jit


class _DistanceBase(metaclass=ABCMeta):
    """
    This is our base distance class.
    """
    @abstractmethod
    def calc_dist(self, mat1, mat2):
        pass


class _PowerDistance(_DistanceBase):
    """
    This is the distance class that uses the p-norm to generate our distances.
    """
    def __init__(self, p):
        self.p = p

    @staticmethod
    @jit(nopython=True)
    def _calc_dist(mat1, mat2, p):
        """

        :param mat1: Sample 1
        :param mat2: Sample 2
        :param p: Our power value for the p-norm
        :return:
        """
        abs_dist = np.abs(mat1.reshape(mat1.shape + (1,)) - mat2.reshape(mat2.shape + (1,)).T) ** p

        return np.sum(abs_dist, axis=1) ** (1.0 / p)

    def calc_dist(self, mat1, mat2):
        """
        This calculates the distance

        :param mat1: Sample 1
        :type mat1: numpy.array
        :param mat2: Sample 2
        :type mat2: numpy.array
        :return:
        """
        return self._calc_dist(mat1, mat2, self.p)


class _MahalanobisDistance(_PowerDistance):

    def __init__(self):
        _PowerDistance.__init__(self, 2.0)

    def calc_dist(self, mat1, mat2):
        """

        :param mat1:
        :param mat2:
        :return:
        """

        # we calculate the covariance matrix
        cov = np.cov(np.vstack((mat1, mat2)).T)

        # we compute the eigenvalue decomposition for symmetric matrices
        x, v = np.linalg.eigh(cov)

        # we want to squash small eigenvalues and only big eigenvalues
        indices = v < 1e-10

        # we decorrelate our matrix, and scale
        mat1_new = np.dot(np.dot(mat1, x[:, ~indices]), np.diag(v[~indices] ** -1.0))
        mat2_new = np.dot(np.dot(mat2, x[:, ~indices]), np.diag(v[~indices] ** -1.0))

        return _PowerDistance.calc_dist(self, mat1_new, mat2_new)


class _CosineDistance(_DistanceBase):
    """
    This uses the cosine distance.
    """
    @staticmethod
    @jit(nopython=True)
    def _calc_dist(mat1, mat2):
        """
        This is a jit function that to help speed up the distance calculation.

        :param mat1: Sample 1
        :param mat2: Sample 2
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
        """
        This calculates the distance

        :param mat1: Sample 1
        :type mat1: numpy.array
        :param mat2: Sample 2
        :type mat2: numpy.array
        :return:
        """
        return self._calc_dist(mat1, mat2)


class _KNNBase(_DistanceBase, ABC):
    """
    This is another abstract class for our k-nearest neighbors algorithm
    """
    def __init__(self, k):
        self.k = k

    @staticmethod
    @jit(nopython=True)
    def _get_k_neighbors(dist, k):
        """

        :param dist: the kernel distance matrix
        :param k: the number of neighbors we want
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
        This is where we calculate the distances, and then perform the match on a batch

        :param mat1_batch: Batch of sample 1
        :param mat2_batch: Batch of sample 2
        :param k: he number of neighbors we want
        :return:
        """

        dist = self.calc_dist(mat1_batch, mat2_batch)

        return self._get_k_neighbors(dist, k)

    def match(self, mat1, mat2):
        """
        Given the memory-related issues with the algorithm.  We create a batch process and calculate the distances
        on a batch of our samples, and generate our matches on that batch.

        :param mat1: Sample 1
        :type mat1: numpy.array
        :param mat2: Sample 2
        :type mat2: numpy.array
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


class KNNPowerMatcher(_PowerDistance, _KNNBase):
    """
    This is the K-Nearest Neighbor algorithm with the p-norm distance measure.
    """
    def __init__(self, k, p):
        _PowerDistance.__init__(self, p)
        _KNNBase.__init__(self, k)


class KNNMahalanobisMatcher(_MahalanobisDistance, _KNNBase):
    """
    This is the K-Nearest Neighbor algorithm with the mahalanobis distance measure.
    """
    def __init__(self, k):
        _KNNBase.__init__(self, k)


class KNNCosineMatcher(_CosineDistance, _KNNBase):
    """
    This is the K-Nearest Neighbor algorithm with the cosine distance measure.
    """
    def __init__(self, k):
        _KNNBase.__init__(self, k)


class _NNLinearSumBase(_DistanceBase, ABC):
    """
    This is the abstract class for the Hungarian Matching Algorithm, where we use the algorithm to exhaust all the
    unique matches between our samples

    """
    def __init__(self):
        super().__init__()

    def match(self, mat1, mat2):
        """
        Get all samples in mat2 to match to mat1 by using the linear_sum_assignment.

        :param mat1: Sample 1
        :type mat1: numpy.array
        :param mat2: Sample 2
        :type mat2: numpy.array
        :return:
        """

        dist = self.calc_dist(mat1, mat2)

        match_lst = []

        n = mat1.shape[0]
        m = mat2.shape[0]

        mat_on_indcs = np.array([True] * n)
        mat1_indices = np.arange(n)

        while True:
            dist_batch = dist[mat1_indices[mat_on_indcs], :]

            # if we have less samples in the first array than second, we break
            if dist_batch.shape[0] <= m:
                cost, x_indices, y_indices = lapjv(dist_batch, extend_cost=True)

                matches = np.column_stack((mat1_indices[mat_on_indcs].reshape(-1, 1),
                                           x_indices.reshape(-1, 1)))

                match_lst.append(matches)

                break

            else:
                # we get all of the possible matches
                cost, x_indices, y_indices = lapjv(dist_batch.T, extend_cost=True)

                # we reset the x_indices for convenience
                x_indices = mat1_indices[mat_on_indcs][x_indices]

                # we get our matches
                matches = np.column_stack((x_indices,
                                           np.arange(m).reshape(-1, 1)))

                # all that were matched already are set to False
                mat_on_indcs[np.isin(mat1_indices, x_indices)] = False

                # we append our matches to the master list
                match_lst.append(matches)

        matches = np.vstack(match_lst)

        return matches[np.argsort(matches[:, 0]), :]


class NNLinearSumPowerMatcher(_PowerDistance, _NNLinearSumBase):
    """
    This is the Exhaustive-Hungarian Matching algorithm with the p-norm distance measure.
    """
    def __init__(self, p):
        _PowerDistance.__init__(self, p)


class NNLinearSumMahalanobisMatcher(_MahalanobisDistance, _NNLinearSumBase):
    """
    This is the Exhaustive-Hungarian Matching algorithm with the mahalanobis distance measure.
    """
    pass


class NNLinearSumCosineMatcher(_CosineDistance, _NNLinearSumBase):
    """
    This is the Exhaustive-Hungarian Matching algorithm with the cosine distance measure.
    """
    pass
