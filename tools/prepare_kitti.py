import argparse
import io
import os
import shutil
from zipfile import ZipFile

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="./data/kitti/raw/")
parser.add_argument("--output_root", type=str, default="./data/kitti/processed/")
parser.add_argument("--make_tar", action="store_true")
args = parser.parse_args()

os.makedirs(args.output_root, exist_ok=True)


def extract_init_seed(pts_sort, n_lpr, th_seed):
    lpr = np.mean(pts_sort[:n_lpr, 2])
    seed = pts_sort[pts_sort[:, 2] < lpr + th_seed, :]
    return seed


def extract_ground(pts):
    th_seeds_ = 1.2
    num_lpr_ = 20
    n_iter = 10
    th_dist_ = 0.3
    pts_sort = pts[pts[:, 2].argsort(), :]
    pts_g = extract_init_seed(pts_sort, num_lpr_, th_seeds_)
    normal_ = np.zeros(3)
    for i in range(n_iter):
        mean = np.mean(pts_g, axis=0)[:3]
        xx = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 0] - mean[0]))
        xy = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 1] - mean[1]))
        xz = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 2] - mean[2]))
        yy = np.mean((pts_g[:, 1] - mean[1]) * (pts_g[:, 1] - mean[1]))
        yz = np.mean((pts_g[:, 1] - mean[1]) * (pts_g[:, 2] - mean[2]))
        zz = np.mean((pts_g[:, 2] - mean[2]) * (pts_g[:, 2] - mean[2]))
        cov = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        U, S, V = np.linalg.svd(cov)
        normal_ = U[:, 2]
        d_ = -normal_.dot(mean)
        th_dist_d_ = th_dist_ - d_
        result = pts[:, :3].dot(normal_)
        pts_n_g = pts[result > th_dist_d_]
        pts_g = pts[result < th_dist_d_]
    return pts_g, pts_n_g


def process_scene(split="training", scene_id="0019"):
    velodyne_path = os.path.join(args.data_root, "data_tracking_velodyne.zip")
    label_path = os.path.join(args.data_root, "data_tracking_label_2.zip")
    calib_path = os.path.join(args.data_root, "data_tracking_calib.zip")
    detection_path = os.path.join(args.data_root, "data_tracking_detection.zip")
    assert os.path.exists(velodyne_path), "Please download the KITTI dataset first"
    assert os.path.exists(label_path), "Please download the KITTI dataset first"
    assert os.path.exists(calib_path), "Please download the KITTI dataset first"
    assert os.path.exists(detection_path), "Please download the KITTI dataset first"

    clean_pcs_dir = os.path.join(args.output_root, split, "clean_pcs", scene_id)
    os.makedirs(clean_pcs_dir, exist_ok=True)

    with ZipFile(velodyne_path) as velodyne_zf:
        scene_files = [
            file for file in velodyne_zf.namelist() if f"{split}/velodyne/{scene_id}" in file and file.endswith(".bin")
        ]
        scene_files = sorted(scene_files)

        for frame_idx, frame_file in enumerate(scene_files):
            assert frame_file == f"{split}/velodyne/{scene_id}/{frame_idx:06d}.bin"
            with io.BufferedReader(velodyne_zf.open(frame_file, mode="r")) as f:
                pc = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 4)
                ground_pc, clean_pc = extract_ground(pc[:, :3])
                np.save(os.path.join(clean_pcs_dir, "{}.npy".format(frame_idx)), clean_pc)

    with ZipFile(detection_path) as detection_zf:
        scene_files = [
            file
            for file in detection_zf.namelist()
            if f"{split}/detection/{scene_id}" in file and file.endswith(".pkl")
        ]
        scene_files = sorted(scene_files)

        for frame_idx, frame_file in enumerate(scene_files):
            assert frame_file == f"{split}/detection/{scene_id}/{frame_idx:06d}.pkl"
            detection_zf.extract(frame_file, path=args.output_root)

    with ZipFile(label_path) as label_zf:
        scene_file = f"{split}/label_02/{scene_id}.txt"
        label_zf.extract(scene_file, path=args.output_root)

    with ZipFile(calib_path) as calib_zf:
        scene_file = f"{split}/calib/{scene_id}.txt"
        calib_zf.extract(scene_file, path=args.output_root)


if __name__ == "__main__":
    process_scene("training", "0019")
    process_scene("training", "0020")
    if args.make_tar:
        shutil.make_archive(
            args.output_root,
            "tar",
            os.path.join(args.output_root, ".."),
            os.path.split(os.path.normpath(args.output_root))[-1],
        )
