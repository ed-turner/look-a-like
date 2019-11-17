# standard python library
from abc import ABCMeta, abstractmethod, ABC
import itertools
import warnings

# numeric package
import numpy as np

# helps speed up performance
from numba import jit

# optimization packages
from ortools.linear_solver import pywraplp
from lap import lapjv



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
        p = self.p

        return self._calc_dist(mat1, mat2, p)


class _MahalanobisDistance(_DistanceBase):

    def __init__(self):
        pass

    def calc_dist(self, mat1, mat2):
        """

        :param mat1:
        :param mat2:
        :return:
        """
        p = 2.0

        # we calculate the covariance matrix
        cov = np.cov(np.vstack((mat1, mat2)).T)

        # we compute the eigenvalue decomposition for symmetric matrices
        x, v = np.linalg.eigh(cov)

        # we want to squash small eigenvalues and only big eigenvalues
        indices = x < 1e-10

        # we decorrelate our matrix, and scale
        mat1_new = np.dot(np.dot(mat1, v[:, ~indices]), np.diag(x[~indices] ** -0.5))
        mat2_new = np.dot(np.dot(mat2, v[:, ~indices]), np.diag(x[~indices] ** -0.5))

        abs_dist = np.abs(mat1_new.reshape(mat1_new.shape + (1,)) - mat2_new.reshape(mat2_new.shape + (1,)).T) ** p

        return np.sum(abs_dist, axis=1) ** (1.0 / p)


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

        k = self.k

        res_lst = []

        for i in range(0, n1, 100):
            tmp_i_indices = np.arange(i, min(i + 100, n1))

            mat1_batch = np.ascontiguousarray(mat1[tmp_i_indices, :])

            unique_i_batch_lst = []

            for j in range(0, n2, 100):
                tmp_j_indices = np.arange(j, min(j + 100, n2))

                mat2_batch = np.ascontiguousarray(mat1[tmp_j_indices, :])

                indices = self._knn_match_batch(mat1_batch, mat2_batch, k)

                for i2 in range(indices.shape[0]):
                    unique_i_batch_lst.append(tmp_j_indices[indices[i2].astype(np.int64)].reshape(1, k))

            unique_mat2_batch_flat = np.unique(
                np.vstack(unique_i_batch_lst).reshape(-1, ).astype(np.int64)
            )

            mat2_batch = mat2[unique_mat2_batch_flat, :]

            tmp_indices = self._knn_match_batch(mat1_batch, mat2_batch, k)

            for z in range(tmp_indices.shape[0]):
                k_indices = np.ascontiguousarray(tmp_indices[z, :].reshape(-1, )).astype(np.int64)

                res_lst.append(np.ascontiguousarray(unique_mat2_batch_flat[k_indices].reshape(-1, k)))

        return np.vstack(res_lst).astype(np.int64)


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


class _EMDMatcher(_DistanceBase, ABC):
    """
    This idea of matching is based on the Earth Mover Distance.

    Please consult this reference: https://en.wikipedia.org/wiki/Earth_mover%27s_distance
    """

    def __init__(self, training_weights, testing_weights, thrsh=1e-14):
        """

        :param thrsh:
        """
        self.wt1 = testing_weights
        self.wt2 = training_weights
        self.thrsh = thrsh

    @staticmethod
    def _batch_match(cost, wt1, wt2, eps):
        """

        :param cost:
        :param wt1:
        :param wt2:
        :param eps: The tolerance threshold
        :return:
        """
        nr = cost.shape[0]
        nd = cost.shape[1]

        adj = wt1.sum() / wt2.sum()
        new_wt2 = wt2 * adj

        solver = pywraplp.Solver('emd_program', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

        def create_var(i, j):
            return solver.NumVar(0., 1, name="row_{}_col_{}".format(i, j))

        indices = itertools.product(range(nr), range(nd))

        graph_vars = {x: create_var(*x) for x in indices}

        for row in range(nr):
            # we assert the percent difference is more than zero
            solver.Add(solver.Sum([graph_vars[row, col] * new_wt2[col] for col in range(nd)]) == wt1[row])

        for col in range(nd):
            # we assert that each training sample is matched at most once
            solver.Add(solver.Sum([graph_vars[row, col] for row in range(nr)]) <= 1)

        obj = solver.Sum([cost[x] * graph_vars[x] for x in indices])

        solver.Minimize(obj)

        result_status = solver.Solve()

        if result_status == pywraplp.Solver.OPTIMAL:
            pass
        else:
            warnings.warn("The solver failed to find an optimized... Results may vary")

        _vals = [list(x) for x in indices if eps < graph_vars[x].solution_value()]

        return np.array(_vals).astype(np.int64)

    def match(self, mat1, mat2):
        """
        Given the training matrix and the testing matrix, along with the sample weights for each of the samples,
        which is suppose to measure their relevance to the universe, we match the samples together such that
        it minimizes the total distance of the matched samples

        :param mat1: The training dataset
        :type mat1: numpy.array
        :param mat2: The testing dataset
        :type mat2: numpy.array
        :return:
        """

        cost = self.calc_dist(mat1, mat2)
        thrsh = self.thrsh
        wt1 = self.wt1
        wt2 = self.wt2

        assert cost.shape[0] == wt2.shape[0]
        assert cost.shape[1] == wt1.shape[0]

        sol = self._batch_match(cost, wt1, wt2, thrsh)

        return sol


class EMDPowerMatcher(_PowerDistance, _EMDMatcher):
    """
    This is the Earth Mover Distance Matching algorithm with the p-norm distance measure.
    """
    def __init__(self, p, training_weights, testing_weights, thrsh):
        _PowerDistance.__init__(self, p)
        _EMDMatcher.__init__(self, training_weights, testing_weights, thrsh)


class EMDMahalanobisMatcher(_MahalanobisDistance, _EMDMatcher):
    """
    This is the Earth Mover Distance Matching algorithm with the mahalanobis distance measure.
    """
    def __init__(self, training_weights, testing_weights, thrsh):
        _EMDMatcher.__init__(self, training_weights, testing_weights, thrsh)


class EMDCosineMatcher(_CosineDistance, _EMDMatcher):
    """
    This is the Earth Mover Distance Matching algorithm with the cosine distance measure.
    """

    def __init__(self, training_weights, testing_weights, thrsh):
        _EMDMatcher.__init__(self, training_weights, testing_weights, thrsh)
