import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import copy
import pandas as pd

"""
Calculation of portfolio weights using HRP methods
"""


def get_corr(price_data):
    return_data = price_data.pct_change().dropna()
    corr = return_data.corr()
    cov = return_data.cov()

    assert False not in corr.isnull(), 'Correlation matrix has nan'

    return corr, cov


def get_ann_return(price_data):
    ann_return = price_data.pct_change().mean() * 252
    return ann_return


def get_distance(corr_matrix):
    dist_matrix = np.sqrt((1 - corr_matrix) / 2)
    return dist_matrix


def seriation(Z, N, cur_index):
    """
    Returns the order implied by a hierarchical tree (dendrogram).
    """

    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))


def compute_serial_matrix(dist_mat, method="single"):
    """Returns a sorted distance matrix.

       :param dist_mat: A distance matrix.
       :param method: A string in ["ward", "single", "average", "complete"].
    """
    dist_mat = dist_mat.values
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)

    res_linkage = linkage(flat_dist_mat, method=method)
    res_order = seriation(res_linkage, N, N + N - 2)

    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)

    seriated_dist[a, b] = dist_mat[[res_order[i]
                                    for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage


def compute_HRP_weights(covariances, res_order):
    weights = pd.Series(1, index=res_order)
    clustered_lists = [res_order]
    while len(clustered_lists) > 0:
        # Bisect：divide clustered_lists into two lists
        clustered_lists = [cluster[start:end] for cluster in clustered_lists
                           for start, end in ((0, len(cluster) // 2),
                                              (len(cluster) // 2, len(cluster)))
                           if len(cluster) > 1]

        for subcluster in range(0, len(clustered_lists), 2):
            left_cluster = clustered_lists[
                subcluster]  # divide into groups every two lists; take the left cluster (list)
            # take the right cluster (list)
            right_cluster = clustered_lists[subcluster + 1]

            left_subcovar = covariances.iloc[
                left_cluster, left_cluster]  # the covariance matrix of the indexes in left clusters
            inv_diag = 1 / np.diag(left_subcovar.values)

            parity_w = inv_diag * (1 / np.sum(inv_diag))
            left_cluster_var = np.dot(
                parity_w, np.dot(
                    left_subcovar, parity_w))  # a value

            right_subcovar = covariances.iloc[
                right_cluster, right_cluster]  # the covariance matrix of the indexes in right clusters
            inv_diag = 1 / np.diag(right_subcovar.values)
            parity_w = inv_diag * (1 / np.sum(inv_diag))
            right_cluster_var = np.dot(
                parity_w, np.dot(
                    right_subcovar, parity_w))

            alloc_factor = 1 - left_cluster_var / \
                (left_cluster_var + right_cluster_var)

            # update all the values contained in left_cluster in weights
            weights[left_cluster] *= alloc_factor
            weights[right_cluster] *= 1 - alloc_factor
    return weights


def get_HRP_result(price_data):
    corr, cov = get_corr(price_data)
    dist = get_distance(corr)
    seriated_dist, res_order, res_linkage = compute_serial_matrix(dist)
    weights_HRP = compute_HRP_weights(cov, res_order)

    return weights_HRP.sort_index(), corr, cov, dist, res_order, res_linkage


def portfolio_reuslt(price_data):

    ann_return = get_ann_return(price_data)
    weights_HRP, corr, cov, dist, res_order, res_linkage = get_HRP_result(
        price_data)

    ann_return = ann_return.values.reshape(-1, 1)
    weights = weights_HRP.values.reshape(-1, 1)
    cov = cov.values

    port_return = ann_return.T.dot(weights)
    port_cov = np.dot(weights.T.dot(cov), weights)
    return port_return, np.sqrt(port_cov * 252), weights_HRP


if __name__ == '__main__':
    price_data = pd.read_csv('nasdaq_100.csv', index_col=0)
    port_return_in, port_sd_in, weights_HRP = portfolio_reuslt(price_data)
