import argparse
import os
import random

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = str(0)

import numpy as np
import tensorflow.compat.v1 as tf
import webdataset as wds
from google.protobuf.descriptor import FieldDescriptor as FD
from tqdm.contrib.concurrent import process_map
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="./data/waymo/raw/validation/")
parser.add_argument("--output_root", type=str, default="./data/waymo/processed/validation/")
args = parser.parse_args()

os.makedirs(args.output_root, exist_ok=True)


def extract_ground(pts):
    ids = np.arange(len(pts))
    th_seeds_ = 1.2
    pts_g = pts[pts[:, 2] > -1.5 * 1.7]
    ids_g = ids[pts[:, 2] > -1.5 * 1.7]
    z_mean = np.mean(pts_g[:, 2])
    pts_seed = pts_g[pts_g[:, 2] < z_mean + th_seeds_]
    pts_g = pts_seed
    pts_n_g = []
    for i in range(10):
        th_dist_ = 0.35
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
        result = pts_g[:, :3].dot(normal_)
        pts_n_g.append(pts_g[result > th_dist_d_])
        pts_g = pts_g[result < th_dist_d_]
    return pts_g, np.vstack(pts_n_g)


# Takes a ProtoBuf Message obj and converts it to a dict.
def pb2dict(obj):
    return_dict = {}
    for field in obj.DESCRIPTOR.fields:
        if not getattr(obj, field.name):
            continue
        if not field.label == FD.LABEL_REPEATED:
            if not field.type == FD.TYPE_MESSAGE:
                return_dict[field.name] = getattr(obj, field.name)
            else:
                value = pb2dict(getattr(obj, field.name))
                if value:
                    return_dict[field.name] = value
        else:
            if field.type == FD.TYPE_MESSAGE:
                return_dict[field.name] = [pb2dict(v) for v in getattr(obj, field.name)]
            else:
                return_dict[field.name] = [v for v in getattr(obj, field.name)]
    return return_dict


def process_segment(tf_record_filename):
    random.seed(0)
    np.random.seed(0)

    tf_record_path = os.path.join(args.data_root, tf_record_filename)
    segment_name = os.path.splitext(tf_record_filename)[0]
    tf_record = tf.data.TFRecordDataset(tf_record_path, compression_type="")

    output_path = os.path.join(args.output_root, f"{segment_name}.tar")
    with wds.TarWriter(output_path) as sink:
        for frame_idx, record in enumerate(tf_record):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(record.numpy()))

            # extract the annotations
            laser_labels = frame.laser_labels

            ego_motion = np.reshape(np.array(frame.pose.transform), [4, 4]).tolist()

            processed_labels = []
            for laser_label in laser_labels:
                id = laser_label.id
                box = laser_label.box
                meta_data = laser_label.metadata

                box_dict = pb2dict(box)
                meta_data_dict = pb2dict(meta_data)

                label = {"id": id, "box": box_dict, "meta_data": meta_data_dict, "type": laser_label.type}
                processed_labels.append(label)

            # extract the points
            (
                range_images,
                camera_projections,
                range_image_top_pose,
            ) = frame_utils.parse_range_image_and_camera_projection(frame)
            points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose, ri_index=0
            )
            all_points = np.concatenate(points, axis=0)

            ground_pc, clean_pc = extract_ground(all_points)

            sample = {
                "__key__": f"{frame_idx:06d}",
                "clean_pc.npy": clean_pc,
                "label.json": processed_labels,
                "ego_motion.json": ego_motion,
            }
            sink.write(sample)


if __name__ == "__main__":
    tf_record_filenames = sorted([x for x in os.listdir(args.data_root) if "tfrecord" in x])
    process_map(process_segment, tf_record_filenames, max_workers=10)
