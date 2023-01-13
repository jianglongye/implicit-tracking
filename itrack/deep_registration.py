import copy
import math
import os
import time

import numpy as np
import torch
import trimesh
from pytorch3d.transforms import Transform3d
from skimage import measure
from torch.nn import functional as F

from itrack import coord
from itrack.loss import chamfer_distance_loss
from itrack.util import setup_logger

WAYMO2SHAPENET_MAT = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
SHAPENET2WAYMO_MAT = np.linalg.inv(WAYMO2SHAPENET_MAT)
WAYMO2SHAPENET_MAT_T = torch.from_numpy(WAYMO2SHAPENET_MAT).float()

SHAPENET_MEAN_SACLE = 0.44


@torch.no_grad()
def encode_pc(pc, model, scale, device=None):
    if device is None:
        device = torch.device("cuda:0")

    # pc = downsample_pc(pc.copy())

    norm_pc = pc / scale * SHAPENET_MEAN_SACLE
    # transform to shapenet coordinate
    norm_pc = coord.transform(WAYMO2SHAPENET_MAT, norm_pc)
    model.eval()
    surface = torch.from_numpy(norm_pc).float().to(device).unsqueeze(0)
    shape_code = model.encode(surface)

    shape_code = shape_code.detach().clone()

    return shape_code


@torch.no_grad()
def get_sdf(pc, model, code, scale, device=None):
    if device is None:
        device = torch.device("cuda:0")

    norm_pc = pc.copy() / scale * SHAPENET_MEAN_SACLE
    # transform to shapenet coordinate
    norm_pc = coord.transform(WAYMO2SHAPENET_MAT, norm_pc)
    model.eval()
    pc = torch.from_numpy(norm_pc).float().unsqueeze(0).to(device)
    pred_sdf = model.decode(pc.permute(0, 2, 1), code)

    return pred_sdf


def finetune_code(pc, shape_code, init_shape_code, model, scale, cfg, device=None):
    if device is None:
        device = torch.device("cuda:0")

    prev_shape_code = shape_code.clone()
    init_shape_code = init_shape_code.clone()
    shape_code.requires_grad = True
    optimizer = torch.optim.SGD([shape_code], lr=cfg.SEARCH_CODE.LR)

    if len(pc) > 2000:
        pc = downsample_pc(pc.copy())

    norm_pc = pc / scale * SHAPENET_MEAN_SACLE
    # transform to shapenet coordinate
    norm_pc = coord.transform(WAYMO2SHAPENET_MAT, norm_pc)
    surface = torch.from_numpy(norm_pc).float().to(device).unsqueeze(0)

    model.eval()
    logger = setup_logger("tune", logging_level="debug" if cfg.DEBUG.IS_DEBUG else "info")
    for i in range(cfg.SEARCH_CODE.ITER_NUM):
        optimizer.zero_grad()
        pred_sdf = model.decode(surface.permute(0, 2, 1), shape_code)
        sdf_loss = torch.abs(pred_sdf).mean()
        prev_reg = torch.abs(shape_code - prev_shape_code).mean() * cfg.SEARCH_CODE.PREV_REG_WEIGHT
        init_reg = torch.abs(shape_code - init_shape_code).mean() * cfg.SEARCH_CODE.INIT_REG_WEIGHT
        zero_reg = torch.abs(shape_code - torch.zeros_like(shape_code)).mean() * cfg.SEARCH_CODE.ZERO_REG_WEIGHT
        loss = sdf_loss + zero_reg + init_reg + prev_reg
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            logger.debug(
                f"%d loss: %.4f, prev_reg: %.4f , init_reg: %.4f, zero reg: %.4f ",
                i,
                loss,
                prev_reg,
                init_reg,
                zero_reg,
            )

        if i != 0 and torch.abs(loss - prev_loss) < prev_loss * 1e-3:
            break
        prev_loss = loss

    shape_code = shape_code.detach().clone()

    return shape_code


def finetune_first_code(pc, shape_code, model, scale, cfg, device=None):
    if device is None:
        device = torch.device("cuda:0")

    shape_code.requires_grad = True
    optimizer = torch.optim.SGD([shape_code], lr=cfg.SEARCH_CODE.LR)

    # pc = downsample_pc(pc.copy())

    norm_pc = pc / scale * SHAPENET_MEAN_SACLE
    # transform to shapenet coordinate
    norm_pc = coord.transform(WAYMO2SHAPENET_MAT, norm_pc)
    surface = torch.from_numpy(norm_pc).float().to(device).unsqueeze(0)

    model.eval()
    logger = setup_logger("tune", logging_level="debug" if cfg.DEBUG.IS_DEBUG else "info")
    for i in range(cfg.SEARCH_CODE.ITER_NUM):
        optimizer.zero_grad()
        pred_sdf = model.decode(surface.permute(0, 2, 1), shape_code)
        sdf_loss = torch.abs(pred_sdf).mean()
        zero_reg = torch.abs(shape_code - torch.zeros_like(shape_code)).mean() * cfg.SEARCH_CODE.ZERO_REG_WEIGHT
        loss = sdf_loss + zero_reg
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            logger.debug(f"%d loss: %.4f, zero_reg: %.4f", i, loss, zero_reg)

        if i != 0 and torch.abs(loss - prev_loss) < prev_loss * 1e-3:
            break
        prev_loss = loss

    shape_code = shape_code.detach().clone()

    return shape_code


def downsample_pc(pc, max_num=2000):
    return pc[torch.randperm(pc.shape[0])][:max_num]


def find_nearest_box(inv_tgt_trans, box_scores, translation, heading, score_thres=0.1, dist_thres=3.0):
    idx = -1
    min_dist = 1e3
    for i, inv_tgt_tran in enumerate(inv_tgt_trans):
        if box_scores[i] < score_thres:
            continue
        dist = np.linalg.norm(translation - inv_tgt_tran[:3])
        if dist < min_dist and dist < dist_thres:
            min_dist = dist
            idx = i
    return idx


def deep_registration(data_dict, model, cfg, device=None):
    if device is None:
        device = torch.device("cuda:0")

    pc0, pc1 = data_dict["pc0"], data_dict["pc1"]
    init_pc, agg_pc = data_dict["init_pc"], data_dict["aggregated_pc"]
    shape_code, scale = data_dict["shape_code"], data_dict["scale"]
    gt_transform, inv_pred_trans_obj_list, = data_dict[
        "gt_transform"
    ], copy.deepcopy(data_dict["inv_pred_trans_obj_list"])
    inv_gt_transform = coord.inv_transformation(gt_transform)

    is_pc_exist = True
    success = True
    if pc0.shape[0] < 10 or pc1.shape[0] < 10:
        is_pc_exist = False

    translation = torch.zeros((1, 3)).to(device)
    rot_z = torch.zeros(1).to(device)

    translation.requires_grad = True
    rot_z.requires_grad = True

    pi_t = torch.Tensor([np.pi]).to(device)
    if cfg.LOSS.USE_CD:
        init_pc_t = torch.from_numpy(init_pc).float().to(device)
        agg_pc_t = torch.from_numpy(agg_pc).float().to(device)
        pc0_t = torch.from_numpy(pc0).float().to(device)
        init_pc_t = downsample_pc(init_pc_t, max_num=cfg.CD_LOSS.MaxPCNum)
        agg_pc_t = downsample_pc(agg_pc_t, max_num=cfg.CD_LOSS.MaxPCNum)
        pc0_t = downsample_pc(pc0_t, max_num=cfg.CD_LOSS.MaxPCNum)

    pc1_t = torch.from_numpy(pc1).float().to(device)
    pc1_t = downsample_pc(pc1_t)
    waymo2shapenet = Transform3d(matrix=WAYMO2SHAPENET_MAT_T.T).to(device)

    if cfg.OPTIM.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD([translation, rot_z], lr=cfg.OPTIM.INIT_LR)
    elif cfg.OPTIM.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam([translation, rot_z], lr=cfg.OPTIM.INIT_LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.OPTIM.MILESTONES, gamma=cfg.OPTIM.GAMMA)

    logger = setup_logger("regist", logging_level="debug" if cfg.DEBUG.IS_DEBUG else "info")
    for i in range(cfg.OPTIM.ITER_NUM):
        begin = time.time()
        transform = Transform3d(device=device).rotate_axis_angle(rot_z / pi_t * 180, "Z").translate(translation)
        input = transform.transform_points(pc1_t)

        loss, cd_loss, shape_loss = 0, 0, 0
        if cfg.LOSS.USE_CD and is_pc_exist:
            if cfg.CD_LOSS.USE_PREV_PC and pc0_t.shape[0] > 0:
                cd0 = chamfer_distance_loss(
                    pc0_t.unsqueeze(0),
                    input.unsqueeze(0),
                    cfg.CD_LOSS.PREV_PC_DIR,
                    cfg.LOSS.CD_LOSS_TYPE,
                    cfg.LOSS.CD_SMOOTH_L1_LOSS_BETA,
                )
                cd_loss += cd0 * cfg.CD_LOSS.PREV_PC_WEIGHT
            if cfg.CD_LOSS.USE_INIT_PC and init_pc_t.shape[0] > 0:
                cd1 = chamfer_distance_loss(
                    init_pc_t.unsqueeze(0),
                    input.unsqueeze(0),
                    cfg.CD_LOSS.INIT_PC_DIR,
                    cfg.LOSS.CD_LOSS_TYPE,
                    cfg.LOSS.CD_SMOOTH_L1_LOSS_BETA,
                )
                cd_loss += cd1 * cfg.CD_LOSS.INIT_PC_WEIGHT
            if cfg.CD_LOSS.USE_AGG_PC and agg_pc_t.shape[0] > 0:
                cd2 = chamfer_distance_loss(
                    agg_pc_t.unsqueeze(0),
                    input.unsqueeze(0),
                    cfg.CD_LOSS.AGG_PC_DIR,
                    cfg.LOSS.CD_LOSS_TYPE,
                    cfg.LOSS.CD_SMOOTH_L1_LOSS_BETA,
                )
                cd_loss += cd2 * cfg.CD_LOSS.AGG_PC_WEIGHT
                loss += cd_loss * cfg.LOSS.CD_WEIGHT

        if cfg.LOSS.USE_SHAPE and is_pc_exist:
            # normalize
            input = input / scale * SHAPENET_MEAN_SACLE
            # transform to shapenet coordinate
            input = waymo2shapenet.transform_points(input)
            input = input.unsqueeze(0)

            pred_sdf = model.decode(input.permute(0, 2, 1), shape_code)

            if cfg.LOSS.SHAPE_LOSS_TYPE == "l2":
                shape_loss = F.mse_loss(pred_sdf, torch.zeros_like(pred_sdf))
            elif cfg.LOSS.SHAPE_LOSS_TYPE == "l1":
                shape_loss = F.l1_loss(pred_sdf, torch.zeros_like(pred_sdf))
            elif cfg.LOSS.SHAPE_LOSS_TYPE == "smooth_l1":
                shape_loss = F.smooth_l1_loss(pred_sdf, torch.zeros_like(pred_sdf), beta=0.05)

            loss += shape_loss * cfg.LOSS.SHAPE_LOSS_WEIGHT

        if cfg.LOSS.USE_MC:
            # motion consistency
            delta_x, delta_y = translation[0, 0:1], translation[0, 1:2]
            v = torch.norm(translation[0, 0:2])
            mc = v * torch.cos((rot_z) / 2) + delta_x
            mc_loss = F.mse_loss(mc, torch.zeros_like(mc))
            loss += mc_loss

        if cfg.LOSS.USE_MP and len(inv_pred_trans_obj_list) > 0:
            # motion prior
            avg_trans = np.stack(inv_pred_trans_obj_list).mean(axis=0)
            prev_trans = inv_pred_trans_obj_list[-1]
            trans_prior = prev_trans * 0.5 + avg_trans * 0.5
            translation_prior = torch.from_numpy(trans_prior[:3]).float().to(device)
            rot_z_prior = torch.from_numpy(trans_prior[3:]).float().to(device)
            mp_loss = F.l1_loss(translation_prior, translation[0]) + F.l1_loss(rot_z_prior, rot_z)
            loss += mp_loss * 0.05

        if cfg.EXP.USE_DETECTION:
            nearest_box_idx = find_nearest_box(
                data_dict["inv_tgt_trans"],
                data_dict["det_scores"],
                translation.detach().cpu().numpy(),
                rot_z.detach().cpu().numpy(),
            )
            if nearest_box_idx != -1:
                inv_tgt_trans = data_dict["inv_tgt_trans"][nearest_box_idx]

                if inv_tgt_trans[3] > np.pi / 2:
                    inv_tgt_trans[3] -= np.pi
                    inv_tgt_trans[:3] = coord.ROT_Z_PI @ inv_tgt_trans[:3]
                elif inv_tgt_trans[3] < -np.pi / 2:
                    inv_tgt_trans[3] += np.pi
                    inv_tgt_trans[:3] = coord.ROT_Z_PI @ inv_tgt_trans[:3]

                tgt_translation = torch.from_numpy(inv_tgt_trans[:3]).float().unsqueeze(0).to(device)
                tgt_heading = torch.from_numpy(inv_tgt_trans[3:]).float().to(device)

                detection_loss = F.l1_loss(translation, tgt_translation) * 3 + F.l1_loss(rot_z, tgt_heading)
                loss += detection_loss

        if isinstance(loss, int):
            success = False
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i != 0 and (torch.abs(loss - prev_loss) < prev_loss * 1e-3 or loss > prev_loss * 1.2):
            for g in optimizer.param_groups:
                if g["lr"] > 0.001:
                    g["lr"] = 0.1 * g["lr"]
                else:
                    g["lr"] = 0.9 * g["lr"]
            if optimizer.param_groups[0]["lr"] < 1e-4 or torch.abs(loss) < 1e-4:
                break

        prev_loss = loss
        if i % 10 == 0:
            logger.debug(
                "%d loss: %.4f, shape loss: %.4f, cd loss: %.4f, cost time: %.4f",
                i,
                loss.data,
                shape_loss * cfg.LOSS.SHAPE_LOSS_WEIGHT,
                cd_loss * cfg.LOSS.CD_WEIGHT if cfg.LOSS.USE_CD else 0.0,
                (time.time() - begin) * 1e3,
            )

    if success:
        obj12obj0 = np.zeros(4)
        obj12obj0[:3] = translation[0].detach().cpu().numpy()
        obj12obj0[3] = rot_z.detach().cpu().numpy()
        result = coord.inv_transformation(obj12obj0)

        if cfg.DEBUG.IS_DEBUG:
            save_prefix = os.path.join(cfg.OUTPUT_DIR, "vis", str(data_dict["id"]), str(data_dict["seq_idx"]))
            result_mat = coord.transformation2mat(result)
            inv_result_mat = np.linalg.inv(result_mat)
            warpped_pc = coord.transform(inv_result_mat, pc1.copy())
            warpped_pc = warpped_pc / scale * SHAPENET_MEAN_SACLE
            warpped_pc = coord.transform(WAYMO2SHAPENET_MAT, warpped_pc)
            norm_pc = pc1.copy() / scale * SHAPENET_MEAN_SACLE
            norm_pc = coord.transform(WAYMO2SHAPENET_MAT, norm_pc)
            prev_pc = pc0.copy() / scale * SHAPENET_MEAN_SACLE
            prev_pc = coord.transform(WAYMO2SHAPENET_MAT, prev_pc)
            init_pc = init_pc.copy() / scale * SHAPENET_MEAN_SACLE
            init_pc = coord.transform(WAYMO2SHAPENET_MAT, init_pc)
            agg_pc = agg_pc.copy() / scale * SHAPENET_MEAN_SACLE
            agg_pc = coord.transform(WAYMO2SHAPENET_MAT, agg_pc)
            trimesh.Trimesh(norm_pc).export(f"{save_prefix}.ply")
            trimesh.Trimesh(init_pc).export(f"{save_prefix}_init.ply")
            trimesh.Trimesh(prev_pc).export(f"{save_prefix}_prev.ply")
            trimesh.Trimesh(agg_pc).export(f"{save_prefix}_agg.ply")
            trimesh.Trimesh(warpped_pc).export(f"{save_prefix}_warp.ply")

            samples = coord.create_grid().contiguous().reshape(1, -1, 3).to(device)
            num_chunks = math.ceil(samples.shape[1] / 5000)
            chunked_points = torch.chunk(samples, chunks=num_chunks, dim=1)
            pred_sdf_list = []
            with torch.no_grad():
                for p in chunked_points:
                    pred_sdf = model.decode(p.permute(0, 2, 1), shape_code)
                    pred_sdf_list.append(pred_sdf)
                pred_sdf = torch.cat(pred_sdf_list, dim=-1)

            verts, faces, normals, values = measure.marching_cubes(
                pred_sdf[0].reshape(64, 64, 64).cpu().numpy(), level=0
            )
            verts = (verts / 63) - 0.5
            trimesh.Trimesh(verts, faces, process=False).export(f"{save_prefix}.obj")

        return {"transform": result, "num_iter": i, "success": True, "inv_transform": obj12obj0}

    else:
        return {"success": False}
