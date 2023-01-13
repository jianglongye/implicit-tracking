import copy
import os
import pickle
import warnings
from math import remainder, tau

import numpy as np
import pandas as pd
import webdataset as wds

from itrack import coord


class WaymoLoader:
    def __init__(self, segment_dir, obj_id, frame_range, return_raw_pc=False):
        self.obj_id = obj_id
        self.frame_range = tuple(frame_range)
        assert return_raw_pc is False, "Not implemented yet"

        wds_dataset = wds.WebDataset(segment_dir + ".tar", nodesplitter=lambda x: x).decode()
        self.wds_iterator = iter(wds_dataset)

        for _ in range(0, self.frame_range[0]):
            next(self.wds_iterator)

        self.__curr = 0
        self.__term = int(self.frame_range[1] - self.frame_range[0] + 1)

    def __iter__(self):
        return self

    def __next__(self):
        if self.__curr >= self.__term:
            raise StopIteration()
        sample = next(self.wds_iterator)

        pc, ego_motion, label = sample["clean_pc.npy"], sample["ego_motion.json"], sample["label.json"]
        ego_motion_mat = np.asarray(ego_motion)
        box = [x for x in label if x["id"] == self.obj_id][0]["box"]

        # transform pc and bbox from vehicle coord to global coord
        pc = coord.transform(ego_motion_mat, pc)
        box = coord.transform_box(ego_motion_mat, box)

        return_dict = {
            "id": self.obj_id,
            "seq_idx": self.__curr,
            "frame_idx": int(sample["__key__"]),
            "gt_bbox": box,
            "pc": pc,
            "ego_motion_mat": ego_motion_mat,
        }

        self.__curr += 1
        return return_dict


class KittiLoader:
    def __init__(self, data_root, id):
        self.id = id
        self.velo_dir = os.path.join(data_root, "velodyne")
        self.detection_dir = os.path.join(data_root, "detection")
        self.clean_pc_dir = os.path.join(data_root, "clean_pcs")
        self.label_dir = os.path.join(data_root, "label_02")
        self.calib_dir = os.path.join(data_root, "calib")

        all_annotations = KittiLoader.load_all_annotations(data_root)

        self.tracklet_ann = copy.deepcopy(all_annotations[id])
        track_id = self.tracklet_ann[0]["track_id"]

        for ann in self.tracklet_ann:
            if ann["track_id"] != track_id:
                warnings.warn("different track id in the tracklet")

        self.__curr = 0
        self.__term = len(self.tracklet_ann)

    def __len__(self):
        return len(self.tracklet_ann)

    def __iter__(self):
        return self

    def __getitem__(self, seq_idx):
        anno = self.tracklet_ann[seq_idx]
        box, pc = self.get_bbox_pc(anno)
        det_boxes, det_scores = self.get_detection(anno)

        return_dict = {
            "id": self.id,
            "seq_idx": seq_idx,
            "frame_idx": anno["frame"],
            "scene": anno["scene"],
            "gt_bbox": box,
            "pc": pc,
            "det_boxes": det_boxes,
            "det_scores": det_scores,
        }

        return return_dict

    def __next__(self):
        if self.__curr >= self.__term:
            raise StopIteration()
        (data, self.__curr) = (self.__getitem__(self.__curr), self.__curr + 1)
        return data

    def get_detection(self, anno):
        detection_path = os.path.join(self.detection_dir, anno["scene"], "{:06}.pkl".format(anno["frame"]))
        with open(detection_path, "rb") as f:
            detection = pickle.load(f)
        boxes_3d = detection[0]["pred_boxes"]

        det_bboxes = []
        for i in range(len(boxes_3d)):
            box_3d = boxes_3d[i]
            bbox = {
                "center_x": box_3d[0],
                "center_y": box_3d[1],
                "center_z": box_3d[2],
                "height": box_3d[4],
                "width": box_3d[5],
                "length": box_3d[3],
                "heading": remainder(box_3d[6], tau),
            }
            det_bboxes.append(bbox)

        scores_3d = detection[0]["pred_scores"].numpy()

        return det_bboxes, scores_3d

    def get_bbox_pc(self, anno):
        calib_path = os.path.join(self.calib_dir, anno["scene"] + ".txt")
        calib = self.read_calib_file(calib_path)
        trans_mat = np.vstack(
            (
                calib["Tr_velo_cam"],
                np.array([0, 0, 0, 1]),
            )
        )

        # velo_path = os.path.join(self.velo_dir, anno['scene'], '{:06}.bin'.format(anno['frame']))
        # velo = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
        # pc = transform(trans_mat, velo[:, :3])

        clean_pc_path = os.path.join(self.clean_pc_dir, anno["scene"], "{}.npy".format(anno["frame"]))
        clean_pc = np.load(clean_pc_path)
        pc = coord.transform(trans_mat, clean_pc[:, :3])

        # pc and bbox are in the opencv coord: x: right, y: down, z: forward
        # we need to make the z-axis point up
        pc = coord.transform(coord.KITTI2WAYMO_MAT, pc)

        bbox = {
            "center_x": anno["z"],
            "center_y": -anno["x"],
            "center_z": anno["height"] / 2 - anno["y"],
            "height": anno["height"],
            "width": anno["width"],
            "length": anno["length"],
            "heading": np.pi / 2 - anno["rotation_y"],
        }
        # use case tracklet_anns[5][5] to validate the sign of the heading

        return bbox, pc

    def read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                values = line.split()
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[values[0]] = np.array([float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    pass
        return data

    @classmethod
    def load_all_annotations(cls, data_root, split="test", category_name="Car"):
        pc_dir = os.path.join(data_root, "clean_pcs")
        label_dir = os.path.join(data_root, "label_02")

        scene_ids = cls.return_scene_ids(split)
        scene_list = []
        for dir in os.listdir(pc_dir):
            if os.path.isdir(os.path.join(pc_dir, dir)) and int(dir) in scene_ids:
                scene_list.append(dir)

        all_annotations = []
        for scene in scene_list:
            label_file = os.path.join(label_dir, scene + ".txt")
            df = pd.read_csv(
                label_file,
                sep=" ",
                names=[
                    "frame",
                    "track_id",
                    "type",
                    "truncated",
                    "occluded",
                    "alpha",
                    "bbox_left",
                    "bbox_top",
                    "bbox_right",
                    "bbox_bottom",
                    "height",
                    "width",
                    "length",
                    "x",
                    "y",
                    "z",
                    "rotation_y",
                ],
            )
            df = df[df["type"] == category_name]
            df.insert(loc=0, column="scene", value=scene)
            for track_id in df.track_id.unique():
                df_tracklet = df[df["track_id"] == track_id]
                df_tracklet = df_tracklet.reset_index(drop=True)
                tracklet_anno = [anno for index, anno in df_tracklet.iterrows()]
                all_annotations.append(tracklet_anno)

        return all_annotations

    @classmethod
    def return_scene_ids(cls, split):
        if "TRAIN" in split.upper():  # Training SET
            scene_ids = list(range(0, 17))
        elif "VALID" in split.upper():  # Validation Set
            scene_ids = list(range(17, 19))
        elif "TEST" in split.upper():  # Testing Set
            scene_ids = list(range(19, 21))
        else:  # Full Dataset
            scene_ids = list(range(21))
        return scene_ids
