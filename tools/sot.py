import argparse
import copy
import json
import os
import pickle
import pprint
import random
import time
import warnings

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

warnings.filterwarnings("ignore", message="R is not a valid rotation matrix")  # numerical issue in pytorch3d
warnings.filterwarnings("ignore", message="invalid value encountered in intersection")

import numpy as np
import torch
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from itrack import coord, metric
from itrack.config import get_default_cfg
from itrack.data import KittiLoader, WaymoLoader
from itrack.deep_registration import (
    deep_registration,
    encode_pc,
    finetune_code,
    finetune_first_code,
)
from itrack.registration import registration
from itrack.shape_model import PointSDFModel
from itrack.util import seed_torch, setup_logger


def deep_sot(id, model, cfg, device=None):
    device = torch.device("cuda:0") if device is None else device
    logger = setup_logger("sot", "debug" if cfg.DEBUG.IS_DEBUG else "info")

    if "kitti" in cfg.DATASET:
        data_loader = KittiLoader(cfg.DATA_DIR, id)
    else:
        with open(cfg.OBJECT_LIST_PATH, "r") as f:
            obj_list = json.load(f)
        obj_info = [x for x in obj_list if x["id"] == id][0]
        segment_dir = os.path.join(cfg.DATA_DIR, obj_info["segment_name"])
        data_loader = WaymoLoader(segment_dir, id, obj_info["frame_range"], return_raw_pc=cfg.EXP.RAW_PC)

    data0 = next(data_loader)
    scale = max(data0["gt_bbox"]["width"], data0["gt_bbox"]["height"], data0["gt_bbox"]["length"]) / 2
    if cfg.DEBUG.IS_DEBUG:
        os.makedirs(os.path.join(cfg.OUTPUT_DIR, "vis", str(id)), exist_ok=True)

    mean_shape_code = np.load(cfg.EXP.MEAN_CODE_PATH)["mean"]
    mean_shape_code = torch.from_numpy(mean_shape_code).float().unsqueeze(0).to(device)

    result_list = []
    pred_trans_obj_list = []  # motion model
    inv_pred_trans_obj_list = []  # inv motion model
    shape_code_dict = {}
    for i, data in enumerate(data_loader):
        data1 = data
        if i == 0:
            pc0_glb = coord.pc_in_box(data0["gt_bbox"], data0["pc"], box_scaling=3)
            obj02glb_mat = coord.box2transform_mat(data0["gt_bbox"])
            glb2obj0_mat = np.linalg.inv(obj02glb_mat)
            pc0_obj0 = coord.transform(glb2obj0_mat, pc0_glb)
            box0_obj0 = coord.transform_box(glb2obj0_mat, data0["gt_bbox"])
            pc1_glb = coord.pc_in_box(data0["gt_bbox"], data1["pc"], box_scaling=3)  # we can't use gt bbox at frame 1
            pc1_obj0 = coord.transform(glb2obj0_mat, pc1_glb)

            if len(pc0_glb) > 10 and len(pc1_glb) > 10:
                pred_trans = registration(pc0_obj0, pc1_obj0)
                pred_trans_mat = coord.transformation2mat(pred_trans)
                # inv_pred_trans = coord.inv_transformation(pred_trans)
                # inv_pred_trans_obj_list.append(inv_pred_trans)
                pred_box_obj0 = coord.transform_box(pred_trans_mat, box0_obj0)
                proposal_box_glb = coord.transform_box(obj02glb_mat, pred_box_obj0)
            else:
                proposal_box_glb = data0["gt_bbox"]
            pred_box_glb = data0["gt_bbox"]

            init_pc_glb = coord.pc_in_box(data0["gt_bbox"], data0["pc"], box_scaling=1)
            init_pc_obj = coord.transform(glb2obj0_mat, init_pc_glb)
            agg_pc_obj = init_pc_obj.copy()
            agg_pc_obj_for_shape = []
            agg_pc_obj_for_shape.append(init_pc_obj.copy())

            if len(init_pc_glb) < 10:
                shape_code = torch.zeros_like(mean_shape_code)
                init_shape_code = torch.zeros_like(mean_shape_code)
            elif cfg.EXP.USE_MEAN_CODE:
                shape_code = mean_shape_code.clone()
                init_shape_code = mean_shape_code.clone()
            else:
                if cfg.EXP.WO_ENCODER:
                    init_shape_code = torch.zeros_like(mean_shape_code)
                else:
                    init_shape_code = encode_pc(init_pc_obj, model, scale, device=device)
                if cfg.SEARCH_CODE.FINETUNE_FIRST_FRAME:
                    init_shape_code = finetune_first_code(
                        init_pc_obj, init_shape_code, model, scale, cfg, device=device
                    )
                shape_code = init_shape_code.clone()
        else:
            if len(pred_trans_obj_list) != 0:
                avg_motion = np.stack(pred_trans_obj_list).mean(axis=0)
                prev_motion = pred_trans_obj_list[-1]
                proposal_trans_obj = (
                    avg_motion * cfg.MOTION_MODEL.AVG_WEIGHT + prev_motion * cfg.MOTION_MODEL.PREV_WEIGHT
                )
                proposal_trans_obj_mat = coord.transformation2mat(proposal_trans_obj)
                proposal_trans_glb = obj02glb_mat @ proposal_trans_obj_mat @ glb2obj0_mat
                proposal_box_glb = coord.transform_box(proposal_trans_glb, pred_box_glb)
            else:
                proposal_box_glb = pred_box_glb

        pc0_glb = coord.pc_in_box(pred_box_glb, data0["pc"])
        obj02glb_mat = coord.box2transform_mat(pred_box_glb)
        glb2obj0_mat = np.linalg.inv(obj02glb_mat)
        pc0_obj0 = coord.transform(glb2obj0_mat, pc0_glb)
        pc1_glb = coord.pc_in_box(proposal_box_glb, data1["pc"])
        pc1_obj0 = coord.transform(glb2obj0_mat, pc1_glb)

        if cfg.EXP.USE_DETECTION:
            det_boxes_obj0 = [coord.transform_box(glb2obj0_mat, copy.deepcopy(box)) for box in data1["det_boxes"]]
            inv_tgt_trans = []
            for box in det_boxes_obj0:
                tgt_trans_mat = coord.box2transform_mat(box)
                inv_tgt_trans_mat = np.linalg.inv(tgt_trans_mat)
                inv_tgt_trans.append(coord.mat2transformation(inv_tgt_trans_mat))

        result = {
            "seq_idx": data1["seq_idx"],
            "frame_idx": data1["frame_idx"],
            "box0": data0["gt_bbox"],
            "box1": data1["gt_bbox"],
            "num_iter": 0,
            "proposal_box": copy.deepcopy(proposal_box_glb),
            "pred_box0": copy.deepcopy(pred_box_glb),
        }

        if (pc0_obj0.shape[0] == 0 or pc1_obj0.shape[0] == 0) and not cfg.EXP.USE_DETECTION:
            pred_box_glb = proposal_box_glb
        else:
            if cfg.SEARCH_CODE.IF_ENCODE:
                shape_code = encode_pc(np.concatenate([pc0_obj0, init_pc_obj]), model, scale, device=device)
            elif cfg.SEARCH_CODE.IF_FINETUNE and i <= cfg.EXP.UPDATE_NUM:
                if cfg.SEARCH_CODE.POLICY == "prev_init":
                    temp_pc = np.concatenate([pc0_obj0, init_pc_obj]).copy()
                elif cfg.SEARCH_CODE.POLICY == "prev":
                    temp_pc = pc0_obj0.copy()
                elif cfg.SEARCH_CODE.POLICY == "agg":
                    temp_pc = agg_pc_obj.copy()
                elif cfg.SEARCH_CODE.POLICY == "random":
                    temp_pc = random.choice(agg_pc_obj_for_shape)
                else:
                    raise KeyError("Unsupported policy")
                if len(temp_pc) > 10:
                    shape_code = finetune_code(temp_pc, shape_code, init_shape_code, model, scale, cfg, device=device)

            gt_box1_obj0 = coord.transform_box(glb2obj0_mat, copy.deepcopy(data1["gt_bbox"]))
            gt_transform = coord.box2transformation(gt_box1_obj0)

            data_dict = {
                "pc0": pc0_obj0,
                "pc1": pc1_obj0,
                "shape_code": shape_code,
                "scale": scale,
                "init_pc": init_pc_obj,
                "aggregated_pc": agg_pc_obj,
                "gt_transform": gt_transform,
                "inv_pred_trans_obj_list": inv_pred_trans_obj_list,
                "id": id,
                "seq_idx": i,
            }

            if cfg.EXP.USE_DETECTION:
                data_dict["det_boxes"] = det_boxes_obj0
                data_dict["inv_tgt_trans"] = inv_tgt_trans
                data_dict["det_scores"] = data1["det_scores"]
            return_dict = deep_registration(data_dict, model, cfg, device=device)
            if return_dict["success"]:

                result["num_iter"] = return_dict["num_iter"]
                pred_trans_obj_list.append(return_dict["transform"])
                inv_pred_trans_obj_list.append(return_dict["inv_transform"])

                pred_trans_obj0_mat = coord.transformation2mat(return_dict["transform"])
                pred_trans_glb_mat = obj02glb_mat @ pred_trans_obj0_mat @ glb2obj0_mat
                pred_box_glb = coord.transform_box(pred_trans_glb_mat, pred_box_glb)

                if i % 5 == 0:
                    obj12glb_mat = coord.box2transform_mat(pred_box_glb)
                    glb2obj1_mat = np.linalg.inv(obj12glb_mat)
                    pc1_glb = coord.pc_in_box(pred_box_glb, data1["pc"], box_scaling=1)
                    pc1_obj1 = coord.transform(glb2obj1_mat, pc1_glb)
                    agg_pc_obj = np.concatenate([agg_pc_obj, pc1_obj1])

                if i % 1 == 0:
                    obj12glb_mat = coord.box2transform_mat(pred_box_glb)
                    glb2obj1_mat = np.linalg.inv(obj12glb_mat)
                    pc1_glb = coord.pc_in_box(pred_box_glb, data1["pc"], box_scaling=1)
                    pc1_obj1 = coord.transform(glb2obj1_mat, pc1_glb)
                    agg_pc_obj_for_shape.append(pc1_obj1)
            else:
                pred_box_glb = proposal_box_glb

        data0 = copy.deepcopy(data1)

        iou, iou_3d = metric.iou3d(pred_box_glb, data1["gt_bbox"])

        shape_code_dict[str(i)] = shape_code.cpu().numpy()
        result["pred_box1"], result["iou_3d"] = copy.deepcopy(pred_box_glb), iou_3d
        result_list.append(copy.deepcopy(result))
        logger.debug("frame #%d  iou 3d: %.4f num iter: %d\n", i, iou_3d, result["num_iter"])

    if result_list:
        logger.debug("mean num iter: %.4f", np.stack([r["num_iter"] for r in result_list]).mean())

    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "summary"), exist_ok=True)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "code"), exist_ok=True)
    with open(os.path.join(cfg.OUTPUT_DIR, "summary", "{}.json".format(id)), "w") as f:
        json.dump(result_list, f)
    with open(os.path.join(cfg.OUTPUT_DIR, "code", "{}.pkl".format(id)), "wb") as f:
        pickle.dump(shape_code_dict, f)

    return result_list


class ObjectListDataset(Dataset):
    def __init__(self, obj_list_path, exp_dir):
        super(ObjectListDataset, self).__init__()
        with open(obj_list_path, "r") as f:
            obj_list = json.load(f)
        summary_dir = os.path.join(exp_dir, "summary")
        exist_ids = [os.path.splitext(f)[0] for f in os.listdir(summary_dir) if "json" in f]
        self.obj_id_list = [o["id"] for o in obj_list if o["id"] not in exist_ids]

    def __len__(self):
        return len(self.obj_id_list)

    def __getitem__(self, index):
        return self.obj_id_list[index]


def setup(rank, world_size, port="10231"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_balanced_index(sorted_tracklet_idx, tracklet_lengths, k):
    assign_num = [0 for _ in range(k)]
    assign_idx = {i: [] for i in range(k)}

    for tracklet_idx in sorted_tracklet_idx:
        insert_idx = assign_num.index(min(assign_num))
        assign_num[insert_idx] += tracklet_lengths[tracklet_idx]
        assign_idx[insert_idx].append(tracklet_idx)
    return assign_idx


def main(rank, device_count, world_size, cfg):
    setup(rank, world_size)
    device_id = rank % device_count
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)

    if "kitti" in cfg.DATASET:
        tracklet_anns = KittiLoader.load_all_annotations(cfg.DATA_DIR, "test")
        tracklet_lengths = [len(ann) for ann in tracklet_anns]
        sorted_tracklet_idx = sorted(range(len(tracklet_lengths)), key=lambda k: tracklet_lengths[k], reverse=True)
        assign_idx = get_balanced_index(sorted_tracklet_idx, tracklet_lengths, world_size)
        id_dataloader = assign_idx[rank]
    else:
        id_dataset = ObjectListDataset(cfg.OBJECT_LIST_PATH, cfg.OUTPUT_DIR)
        sampler = DistributedSampler(id_dataset)
        id_dataloader = DataLoader(id_dataset, batch_size=1, num_workers=0, sampler=sampler)

    model = PointSDFModel(
        cfg.SHAPE_MODEL.CODE_DIM,
        cfg.SHAPE_MODEL.HIDDEN_DIM,
        cfg.SHAPE_MODEL.POINT_FEAT_DIMS,
        cfg.SHAPE_MODEL.DECODER_DIMS,
        cfg.SHAPE_MODEL.USE_RES_DECODER,
    )
    stade_dict = torch.load(cfg.CKPT_PATH, map_location=f"cuda:{device_id}")["model"]
    stade_dict = {k.replace("module.", ""): stade_dict[k] for k in stade_dict}
    model.load_state_dict(stade_dict)
    model = model.to(device)

    for id in id_dataloader:
        begin = time.time()
        try:
            seed_torch()
            if "kitti" in cfg.DATASET:
                id = [id]
            result_list = deep_sot(id[0], model, cfg, device=device)
            if len(result_list) > 0:
                iou_list = [r["iou_3d"] for r in result_list]
                print("id:", id[0], "mean iou:", np.stack(iou_list).mean(), "cost time:", time.time() - begin)
        except Exception as e:
            print("error id:", id[0], "")
            print(e)
    cleanup()


def launch(main_fn, device_count, world_size, cfg):
    mp.spawn(main_fn, args=(device_count, world_size, cfg), nprocs=world_size, join=True)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="3D SOT")
    arg_parser.add_argument("--process-num", default=4, type=int, help="number of processes per gpu")
    arg_parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    arg_parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = arg_parser.parse_args()

    cfg = get_default_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "summary"), exist_ok=True)
    cfg.dump(stream=open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w"))

    logger = setup_logger(name="SOT")
    logger.info(f"Command line arguments: {str(args)}")
    logger.info(f"Contents of args.config_file={args.config_file}:\n{pprint.pformat(cfg, indent=4)}")

    device_count = torch.cuda.device_count()
    world_size = device_count * int(args.process_num)

    if "kitti" not in cfg.DATASET:
        id_dataset = ObjectListDataset(cfg.OBJECT_LIST_PATH, cfg.OUTPUT_DIR)
        world_size = min(len(id_dataset), world_size)
        logger.info(f"device count: {device_count} num instances: {len(id_dataset)} num process: {world_size}")

    launch(main, device_count, world_size, cfg)
