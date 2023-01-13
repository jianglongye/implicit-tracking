import numba
import numpy as np
from scipy import optimize
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors

from itrack import coord


def registration(
    x, y, init_transform=None, max_iter=100, loss_type="l2", min_pc_num=10, max_pc_num=1000, neighbor_num=1
):
    assert x.shape[0] > min_pc_num and y.shape[0] > min_pc_num
    if x.shape[0] > max_pc_num:
        x = x[np.random.choice(x.shape[0], max_pc_num)]
    if y.shape[0] > max_pc_num:
        y = y[np.random.choice(y.shape[0], max_pc_num)]

    nbrs = NearestNeighbors(n_neighbors=neighbor_num, algorithm="kd_tree").fit(y)

    if init_transform is not None:
        transform = init_transform
    else:
        transform = np.zeros(4)
    transform_mat = coord.transformation2mat(transform)

    for i in range(max_iter):
        transformed_x = coord.transform(transform_mat, x)
        _, indices = nbrs.kneighbors(transformed_x)
        indices = np.squeeze(indices, axis=1)
        paired_y = y[indices].copy()

        dist_mat = distance_matrix(transformed_x, y)
        row_ind, col_ind = linear_sum_assignment(dist_mat)
        extra_paired_y = y[indices].copy()
        extra_paired_y[row_ind] = y[col_ind].copy()

        result = optimize.minimize(
            fun=registration_loss,
            x0=transform[:],
            args=(x, paired_y, extra_paired_y, loss_type),
            method="BFGS",
            jac=registration_jac,
            options={"maxiter": 50, "disp": False},
        )
        transform = result.x
        transform_mat = coord.transformation2mat(transform)

    return transform


def registration_loss(transform, pc_a, pc_b, extra_pc_b, loss_type):
    # update to new locations
    point_num = pc_a.shape[0]
    x, y, z, theta = transform
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    new_pcs = rotation_matrix @ (pc_a).T
    new_pcs = new_pcs.T[:, :3]
    new_pcs += np.array([x, y, z])

    # compute the loss
    dist = pc_b - new_pcs
    dist = dist + (extra_pc_b - new_pcs)

    if loss_type == "l2":
        dist = dist * dist
        loss = np.sum(dist) / point_num
    elif loss_type == "huber":
        loss = huber_loss(dist, 0.5)
    elif loss_type == "l1":
        dist = np.abs(dist)
        loss = np.sum(dist) / point_num
    elif loss_type == "sqrt":
        dist = np.sqrt(np.abs(dist) + 0.01)
        loss = np.sum(dist) / point_num

    return loss


def registration_jac(transform, pc_a, pc_b, extra_pc_b, loss_type):
    # update to new locations
    x, y, z, theta = transform
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    new_pcs = rotation_matrix @ (pc_a).T
    new_pcs = new_pcs.T[:, :3]
    new_pcs += np.array([x, y, z])
    dist = pc_b - new_pcs
    dist = dist + (extra_pc_b - new_pcs)

    if loss_type == "l2":
        # Derivative of x, y
        derive = -2 * np.average(dist[:, :3], axis=0)
        derive_x, derive_y, derive_z = derive

        # Derivative of theta
        tmp_transformation = np.array([[np.sin(theta), -np.cos(theta)], [np.cos(theta), np.sin(theta)]])
        derive_theta_pcs = pc_a[:, :2] @ tmp_transformation
        # here we get [N * [xsin + ycos, -xcos + ysin]]
        derive_theta = np.sum(dist[:, :2] * derive_theta_pcs, axis=1)
        derive_theta = np.average(derive_theta)
        derive_theta *= 2
        derivative = np.array([derive_x, derive_y, derive_z, derive_theta])
    elif loss_type == "huber":
        derivative = huber_jac(theta, dist, pc_a, 0.5)
    elif loss_type == "l1":
        sign = np.sign(dist[:, :3])
        derivative = -1 * np.average(sign, axis=0)
        derive_x, derive_y, derive_z = derivative

        mask_xy = sign[:, :2]
        tmp_transformation = np.array([[np.sin(theta), -np.cos(theta)], [np.cos(theta), np.sin(theta)]])
        derive_theta_pcs = pc_a[:, :2] @ tmp_transformation
        derivative_theta = np.sum(mask_xy * derive_theta_pcs, axis=1)
        derive_theta = np.average(derivative_theta)
        derivative = np.array([derive_x, derive_y, derive_z, derive_theta])
    elif loss_type == "sqrt":
        sign = np.sign(dist)
        derivative = -sign * 0.5 / np.sqrt(np.abs(dist) + 0.01)
        xyz_deriv = np.average(derivative[:, :3], axis=0)
        derive_x, derive_y, derive_z = xyz_deriv

        tmp_transformation = np.array([[np.sin(theta), -np.cos(theta)], [np.cos(theta), np.sin(theta)]])
        derive_theta_pcs = pc_a[:, :2] @ tmp_transformation
        derive_theta = np.average(-derive_theta_pcs * derivative[:, :2])
        derivative = np.array([derive_x, derive_y, derive_z, derive_theta])

    return derivative


@numba.njit
def huber_loss(dist, huber_limit):
    result = np.zeros((dist.shape[0], 3), dtype=np.float64)
    for i in range(dist.shape[0]):
        for j in range(3):
            val = dist[i, j]
            if np.abs(val) <= huber_limit:
                result[i, j] = val * val
            else:
                result[i, j] = 2 * huber_limit * (np.abs(val) - huber_limit / 2)
    loss = np.sum(result) / dist.shape[0]
    return loss * (0.5 / huber_limit)


@numba.njit
def huber_jac(theta, dist, pca, huber_limit):
    result = np.zeros((dist.shape[0], 4), dtype=np.float64)

    sin_val = np.sin(theta)
    cos_val = np.cos(theta)

    for i in range(dist.shape[0]):
        # x
        val = dist[i, 0]
        theta_der = sin_val * pca[i, 0] + cos_val * pca[i, 1]
        if np.abs(val) <= huber_limit:
            result[i, 0] = -2 * val
            result[i, 3] += 2 * theta_der * val
        else:
            result[i, 0] = -2 * huber_limit * np.sign(val)
            result[i, 3] += -result[i, 0] * theta_der
        # y
        val = dist[i, 1]
        theta_der = -cos_val * pca[i, 0] + sin_val * pca[i, 1]
        if np.abs(val) <= huber_limit:
            result[i, 1] = -2 * val
            result[i, 3] += 2 * theta_der * val
        else:
            result[i, 1] = -2 * huber_limit * np.sign(val)
            result[i, 3] += -result[i, 1] * theta_der
        # z
        val = dist[i, 2]
        if np.abs(val) <= huber_limit:
            result[i, 2] = -2 * val
        else:
            result[i, 2] = -2 * huber_limit * np.sign(val)

    derivative = np.sum(result, axis=0) / result.shape[0]
    return derivative * (0.5 / huber_limit)
