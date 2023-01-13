import argparse
import json
import os
import warnings

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

warnings.filterwarnings("ignore", message="R is not a valid rotation matrix")  # numerical issue in pytorch3d
warnings.filterwarnings("ignore", message="invalid value encountered in intersection")

import numpy as np

from itrack import metric

ROB_THRESHOLDS = np.arange(0, 1.01, 0.05)


class Success(object):
    """Computes and stores the Success"""

    def __init__(self, n=21, max_overlap=1):
        self.max_overlap = max_overlap
        self.Xaxis = np.linspace(0, self.max_overlap, n)
        self.reset()

    def reset(self):
        self.overlaps = []

    def add_overlap(self, val):
        self.overlaps.append(val)

    @property
    def count(self):
        return len(self.overlaps)

    @property
    def value(self):
        succ = [np.sum([i >= thres for i in self.overlaps]).astype(float) / self.count for thres in self.Xaxis]
        return np.array(succ)

    @property
    def average(self):
        if len(self.overlaps) == 0:
            return 0
        return np.trapz(self.value, x=self.Xaxis) * 100 / self.max_overlap


class Precision(object):
    """Computes and stores the Precision"""

    def __init__(self, n=21, max_accuracy=2):
        self.max_accuracy = max_accuracy
        self.Xaxis = np.linspace(0, self.max_accuracy, n)
        self.reset()

    def reset(self):
        self.accuracies = []

    def add_accuracy(self, val):
        self.accuracies.append(val)

    @property
    def count(self):
        return len(self.accuracies)

    @property
    def value(self):
        prec = [np.sum([i <= thres for i in self.accuracies]).astype(float) / self.count for thres in self.Xaxis]
        return np.array(prec)

    @property
    def average(self):
        if len(self.accuracies) == 0:
            return 0
        return np.trapz(self.value, x=self.Xaxis) * 100 / self.max_accuracy


def tracklet_rob(ious, thresholds):
    """compute the robustness of a tracklet"""

    def compute_area(values):
        """compute the approximate integral"""
        area = sum(values[1:-1])
        area = area + (values[0] + values[-1]) / 2
        area *= 0.05
        return area

    robustness = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        robustness[i] = track_len_ratio(ious, threshold)
    rob = compute_area(robustness)
    return rob


def track_len_ratio(ious, threshold):
    """the ratio of successful tracking, for computing robustness"""
    track_len = -1
    for i, iou in enumerate(ious):
        if iou < threshold:
            track_len = i + 1
            break
    if track_len == -1:
        track_len = len(ious)
    return track_len / (len(ious) + 1e-6)


def extract_iou_from_summary_path(summary_path):
    with open(summary_path, "r") as f:
        summary = json.load(f)

    iou_dict = {}
    for s in summary:
        _, iou = metric.iou3d(s["box1"], s["pred_box1"])
        iou_dict[str(s["frame_idx"])] = iou
    # iou_dict = {str(s['frame_idx']): s['iou_3d'] for s in summary}
    return iou_dict


def extract_acc_from_summary_path(summary_path):
    with open(summary_path, "r") as f:
        summary = json.load(f)

    acc_dict = {}
    for s in summary:
        box_a, box_b = s["box1"], s["pred_box1"]
        center_a = np.array([box_a["center_x"], box_a["center_y"], box_a["center_z"]])
        center_b = np.array([box_b["center_x"], box_b["center_y"], box_b["center_z"]])
        acc_dict[str(s["frame_idx"])] = np.linalg.norm(center_a - center_b, ord=2)
    return acc_dict


def extract_info_from_exp_dir(exp_dir, ids=None):
    all_iou = {}
    all_rob = {}
    all_acc = {}
    all_filenames = [f for f in os.listdir(exp_dir) if os.path.isfile(os.path.join(exp_dir, f))]
    for filename in all_filenames:
        id = os.path.splitext(filename)[0]
        if filename == "specs.json":
            continue
        if ids is not None and id not in ids:
            continue

        iou_dict = extract_iou_from_summary_path(os.path.join(exp_dir, filename))
        acc_dict = extract_acc_from_summary_path(os.path.join(exp_dir, filename))
        all_iou[id] = iou_dict
        all_acc[id] = acc_dict
        all_rob[id] = tracklet_rob(list(iou_dict.values()), ROB_THRESHOLDS)

    assert len(all_iou) == len(all_rob) == len(all_acc)
    sorted_all_iou = dict(sorted(all_iou.items()))
    sorted_all_rob = dict(sorted(all_rob.items()))
    sorted_all_acc = dict(sorted(all_acc.items()))
    return sorted_all_iou, sorted_all_rob, sorted_all_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default="./output/new_bench_agg/summary")
    args = parser.parse_args()

    iou_dict, rob_dict, acc_dict = extract_info_from_exp_dir(args.exp_dir)
    all_length_list = [len(iou_dict[k]) for k in iou_dict]
    all_rob_list = [rob_dict[k] for k in rob_dict]

    all_iou_list, all_acc_list = [], []
    for k in iou_dict:
        all_iou_list.extend(list(iou_dict[k].values()))
        all_acc_list.extend(list(acc_dict[k].values()))

    success0 = Success()
    precision0 = Precision()
    for iou, acc in zip(all_iou_list, all_acc_list):
        success0.add_overlap(iou)
        precision0.add_accuracy(acc)

    rob = sum([l * r for l, r in zip(all_length_list, all_rob_list)])

    print("exp:", args.exp_dir.split("/")[-2])
    print(
        f"acc: {np.stack(all_iou_list).mean():.4f}",
        f"rob: {rob / sum(all_length_list):.4f}",
        f"success: {success0.average:.4f}",
        f"precision: {precision0.average:.4f}",
        f"num instances: {len(iou_dict)}",
    )
    print()
