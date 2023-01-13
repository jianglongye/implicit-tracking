import copy
import math

import numba
import numpy as np
import torch
from skimage import measure

SHAPENET_MEAN_SACLE = 0.44

WAYMO2SHAPENET_MAT = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
WAYMO2SHAPENET_MAT_T = torch.from_numpy(WAYMO2SHAPENET_MAT).float()

SHAPENET2WAYMO_MAT = np.linalg.inv(WAYMO2SHAPENET_MAT)

ROT_Z_PI = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

KITTI2WAYMO_MAT = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

DUMMY_BOX = {
    "center_x": 0.0,
    "center_y": 0.0,
    "center_z": 0.0,
    "heading": 0.0,
    "height": 1.0,
    "width": 1.0,
    "length": 1.0,
}


def transform(transform_mat, pcs):
    pcs = pcs.copy()
    new_pcs = np.concatenate((pcs, np.ones(pcs.shape[0])[:, np.newaxis]), axis=1)
    new_pcs = transform_mat @ new_pcs.T
    new_pcs = new_pcs.T[:, :3]
    return new_pcs.copy()


def array2box(data):
    x, y, z, o, l, w, h = data[:7]
    return {"center_x": x, "center_y": y, "center_z": z, "heading": o, "length": l, "width": w, "height": h}


def transform_box(transform_mat, box):
    # center and corners
    box = copy.deepcopy(box)
    corners = np.array(box2corners2d(box))
    center = np.array([[box["center_x"], box["center_y"], box["center_z"]]])
    center = transform(transform_mat, center)[0]
    corners = transform(transform_mat, corners)
    # heading
    edge_mid_point = (corners[0] + corners[1]) / 2
    yaw = center_edge2yaw(center[:2], edge_mid_point[:2])

    box["center_x"], box["center_y"], box["center_z"] = center[0], center[1], center[2]
    box["heading"] = yaw

    return box


# compute the yaw of objects according to the center and mid-point of edge
def center_edge2yaw(center, edge):
    vec = edge - center
    yaw = np.arccos(vec[0] / np.linalg.norm(vec))
    if vec[1] < 0:
        yaw = -yaw
    return yaw


def box2transform_mat(box):
    x, y, z, theta = box["center_x"], box["center_y"], box["center_z"], box["heading"]
    transform_mat = np.array(
        [[np.cos(theta), -np.sin(theta), 0, x], [np.sin(theta), np.cos(theta), 0, y], [0, 0, 1, z], [0, 0, 0, 1]]
    )
    return transform_mat


def box2transformation(box):
    x, y, z, theta = box["center_x"], box["center_y"], box["center_z"], box["heading"]
    return np.array([x, y, z, theta])


def transformation2mat(trans):
    x, y, z, theta = trans
    transform_mat = np.array(
        [[np.cos(theta), -np.sin(theta), 0, x], [np.sin(theta), np.cos(theta), 0, y], [0, 0, 1, z], [0, 0, 0, 1]]
    )
    return transform_mat


def mat2transformation(mat):
    dummy_box = copy.deepcopy(DUMMY_BOX)
    new_box = transform_box(mat, dummy_box)
    return box2transformation(new_box)


def inv_transformation(trans):
    trans_mat = transformation2mat(trans)
    inv_trans_mat = np.linalg.inv(trans_mat)
    inv_trans = mat2transformation(inv_trans_mat)
    return inv_trans


def box2corners2d(box):
    center_x, center_y, length, width = box["center_x"], box["center_y"], box["length"], box["width"]
    yaw = box["heading"]
    center_z, height = box["center_z"], box["height"]

    center_point = np.array([center_x, center_y, center_z])
    bottom_center = np.array([center_x, center_y, center_z - height / 2])
    pc0 = np.array(
        [
            center_x + np.cos(yaw) * length / 2 + np.sin(yaw) * width / 2,
            center_y + np.sin(yaw) * length / 2 - np.cos(yaw) * width / 2,
            center_z - height / 2,
        ]
    )
    pc1 = np.array(
        [
            center_x + np.cos(yaw) * length / 2 - np.sin(yaw) * width / 2,
            center_y + np.sin(yaw) * length / 2 + np.cos(yaw) * width / 2,
            center_z - height / 2,
        ]
    )
    pc2 = 2 * bottom_center - pc0
    pc3 = 2 * bottom_center - pc1

    return [pc0.tolist(), pc1.tolist(), pc2.tolist(), pc3.tolist()]


def box2corners3d(box):
    center = np.array([box["center_x"], box["center_y"], box["center_z"]])
    bottom_corners = np.array(box2corners2d(box))
    up_corners = 2 * center - bottom_corners
    corners = np.concatenate([up_corners, bottom_corners], axis=0)
    return corners.tolist()


def pc_in_box(box, pc, box_scaling=1.5):
    center_x, center_y, length, width = box["center_x"], box["center_y"], box["length"], box["width"]
    center_z, height = box["center_z"], box["height"]
    yaw = box["heading"]
    return pc_in_box_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling)


def pc_not_in_box(box, pc, box_scaling=1.5):
    center_x, center_y, length, width = box["center_x"], box["center_y"], box["length"], box["width"]
    center_z, height = box["center_z"], box["height"]
    yaw = box["heading"]
    return pc_not_in_box_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling)


def pc_in_box_2D(box, pc, box_scaling=1.0):
    center_x, center_y, length, width = box["center_x"], box["center_y"], box["length"], box["width"]
    center_z, height = box["center_z"], box["height"]
    yaw = box["heading"]
    return pc_in_box_2D_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling)


@numba.njit
def pc_in_box_2D_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling=1.0):
    mask = np.zeros(pc.shape[0], dtype=np.int32)
    yaw_cos, yaw_sin = np.cos(yaw), np.sin(yaw)
    for i in range(pc.shape[0]):
        rx = np.abs((pc[i, 0] - center_x) * yaw_cos + (pc[i, 1] - center_y) * yaw_sin)
        ry = np.abs((pc[i, 0] - center_x) * -yaw_sin + (pc[i, 1] - center_y) * yaw_cos)

        if rx < (length * box_scaling / 2) and ry < (width * box_scaling / 2):
            mask[i] = 1
    indices = np.argwhere(mask == 1)
    result = np.zeros((indices.shape[0], 3), dtype=np.float64)
    for i in range(indices.shape[0]):
        result[i, :] = pc[indices[i], :]
    return result


@numba.njit
def pc_in_box_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling=1.5):
    mask = np.zeros(pc.shape[0], dtype=np.int32)
    yaw_cos, yaw_sin = np.cos(yaw), np.sin(yaw)
    for i in range(pc.shape[0]):
        rx = np.abs((pc[i, 0] - center_x) * yaw_cos + (pc[i, 1] - center_y) * yaw_sin)
        ry = np.abs((pc[i, 0] - center_x) * -yaw_sin + (pc[i, 1] - center_y) * yaw_cos)
        rz = np.abs(pc[i, 2] - center_z)

        if rx < (length * box_scaling / 2) and ry < (width * box_scaling / 2) and rz < (height * box_scaling / 2):
            mask[i] = 1
    indices = np.argwhere(mask == 1)
    result = np.zeros((indices.shape[0], 3), dtype=np.float64)
    for i in range(indices.shape[0]):
        result[i, :] = pc[indices[i], :]
    return result


@numba.njit
def pc_not_in_box_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling=1.5):
    mask = np.zeros(pc.shape[0], dtype=np.int32)
    yaw_cos, yaw_sin = np.cos(yaw), np.sin(yaw)
    for i in range(pc.shape[0]):
        rx = np.abs((pc[i, 0] - center_x) * yaw_cos + (pc[i, 1] - center_y) * yaw_sin)
        ry = np.abs((pc[i, 0] - center_x) * -yaw_sin + (pc[i, 1] - center_y) * yaw_cos)
        rz = np.abs(pc[i, 2] - center_z)

        if rx > (length * box_scaling / 2) or ry > (width * box_scaling / 2) or rz > (height * box_scaling / 2):
            mask[i] = 1
    indices = np.argwhere(mask == 1)
    result = np.zeros((indices.shape[0], 3), dtype=np.float64)
    for i in range(indices.shape[0]):
        result[i, :] = pc[indices[i], :]
    return result


def create_grid(
    voxel_origin=None,
    voxel_size=None,
    resolution=64,
):
    # the voxel_origin is actually the (bottom, left, down) corner, not the center
    if voxel_size is None:
        voxel_size = [1.0, 1.0, 1.0]
    if voxel_origin is None:
        voxel_origin = [-0.5, -0.5, -0.5]

    step_size = [size / (resolution - 1) for size in voxel_size]

    overall_index = torch.arange(0, resolution**3, 1, out=torch.LongTensor())

    samples = torch.zeros(resolution**3, 3).float()

    # transform first 3 columns to be the x, y, z index
    samples[:, 2] = overall_index % resolution
    samples[:, 1] = (overall_index // resolution) % resolution
    samples[:, 0] = ((overall_index // resolution) // resolution) % resolution

    # transform first 3 columns to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * step_size[0]) + voxel_origin[0]
    samples[:, 1] = (samples[:, 1] * step_size[1]) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * step_size[2]) + voxel_origin[2]

    samples.requires_grad = False
    samples = samples.reshape((resolution, resolution, resolution, 3))

    return samples


def create_sections(min=-3, max=3, res=100):
    x = np.linspace(min, max, res)
    y = np.linspace(min, max, res)
    xv, yv = np.meshgrid(x, y)
    sections = np.zeros((res * res, 3))
    sections[:, 0] = xv.reshape(res * res)
    sections[:, 2] = yv.reshape(res * res)
    return sections


def marching_cubes(shape_code, model, device=None, resolution=64):
    if device is None:
        device = torch.device("cuda:0")

    samples = create_grid(resolution=resolution).contiguous().reshape(1, -1, 3).to(device)
    num_chunks = math.ceil(samples.shape[1] / 5000)
    chunked_points = torch.chunk(samples, chunks=num_chunks, dim=1)
    pred_sdf_list = []
    with torch.no_grad():
        for p in chunked_points:
            pred_sdf = model.decode(p.permute(0, 2, 1), shape_code)
            pred_sdf_list.append(pred_sdf)
        pred_sdf = torch.cat(pred_sdf_list, dim=-1)

    verts, faces, normals, values = measure.marching_cubes(
        pred_sdf[0].reshape(resolution, resolution, resolution).cpu().numpy(), level=0
    )
    verts = (verts / (resolution - 1)) - 0.5

    return verts, faces
