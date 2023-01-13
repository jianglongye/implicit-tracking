import matplotlib.pyplot as plt
import numpy as np
import trimesh

from itrack import coord


def save_contour(value, title, output_path, vmin=-3.0, vmax=3.0):
    plt.title(title)
    cset = plt.contourf(value, vmin=vmin, vmax=vmax)
    contour = plt.contour(value, vmin=vmin, vmax=vmax)
    plt.clabel(contour, colors="k")
    plt.xticks(())
    plt.yticks(())
    plt.colorbar(cset)
    plt.savefig(output_path)
    plt.clf()


box_faces = np.array(
    [
        [1, 2, 3],
        [1, 4, 2],
        [1, 3, 0],
        [1, 7, 4],
        [2, 4, 3],
        [4, 5, 3],
        [0, 3, 5],
        [0, 5, 6],
        [1, 0, 7],
        [7, 0, 6],
        [4, 7, 5],
        [5, 7, 6],
    ]
)


def box_to_mesh(box, only_corners=False):
    if not isinstance(box, list):
        box = [box]

    box_mesh_list = []
    box_corners_list = []
    for b in box:
        box_corners = coord.box2corners3d(b)
        box_mesh = trimesh.Trimesh(box_corners, box_faces, process=False)
        box_mesh_list.append(box_mesh)
        if only_corners:
            box_corners_list.append(box_corners)

    box_mesh = trimesh.util.concatenate(box_mesh_list)
    if only_corners:
        box_mesh = trimesh.Trimesh(np.concatenate(box_corners_list), process=False)

    return box_mesh


class Visualizer2d:
    def __init__(self, figsize=(20, 20), title=None):
        self.figure = plt.figure(figsize=figsize)
        self.title = title
        self.figsize = figsize

        plt.axis("equal")
        self.COLOR_MAP = {
            "pc": np.array([140, 140, 136]) / 256,
            "proposal_bbox": np.array([0, 0, 0]) / 256,
            "gt_bbox": np.array([4, 157, 217]) / 256,
            "pred_bbox": np.array([191, 4, 54]) / 256,
        }
        if title is not None:
            plt.title(title, fontsize=figsize[0] * 2)

    def show(self):
        plt.show()

    def close(self):
        plt.close()

    def save(self, path):
        plt.savefig(path)

    def handle_paired_gt_bbox(self, gt_bboxa, gt_bboxb):
        gt_cornersa = np.array(coord.box2corners2d(gt_bboxa))[:, :2]
        gt_cornersb = np.array(coord.box2corners2d(gt_bboxb))[:, :2]

        # so that plt.plot can form a closed rectangle
        gt_cornersa = np.concatenate([gt_cornersa, gt_cornersa[0:1, :]])
        gt_cornersb = np.concatenate([gt_cornersb, gt_cornersb[0:1, :]])

        plt.plot(gt_cornersa[:, 0], gt_cornersa[:, 1], color=self.COLOR_MAP["gt_bbox"], linestyle=":")
        plt.plot(gt_cornersb[:, 0], gt_cornersb[:, 1], color=self.COLOR_MAP["gt_bbox"], linestyle="-")

    def handle_paired_pred_bbox(self, pred_bboxa, pred_bboxb):
        pred_cornersa = np.array(coord.box2corners2d(pred_bboxa))[:, :2]
        pred_cornersb = np.array(coord.box2corners2d(pred_bboxb))[:, :2]

        # so that plt.plot can form a closed rectangle
        pred_cornersa = np.concatenate([pred_cornersa, pred_cornersa[0:1, :]])
        pred_cornersb = np.concatenate([pred_cornersb, pred_cornersb[0:1, :]])

        plt.plot(pred_cornersa[:, 0], pred_cornersa[:, 1], color=self.COLOR_MAP["pred_bbox"], linestyle=":")
        plt.plot(pred_cornersb[:, 0], pred_cornersb[:, 1], color=self.COLOR_MAP["pred_bbox"], linestyle="-")

    def handle_pred_bbox(self, pred_bbox):
        pred_corners = np.array(coord.box2corners2d(pred_bbox))[:, :2]
        pred_corners = np.concatenate([pred_corners, pred_corners[0:1, :]])
        plt.plot(pred_corners[:, 0], pred_corners[:, 1], color=self.COLOR_MAP["pred_bbox"], linestyle="-")

    def handle_paired_pc(self, pca, pcb):
        plt.scatter(pca[:, 0], pca[:, 1], marker="x", color=self.COLOR_MAP["pc"])
        plt.scatter(pcb[:, 0], pcb[:, 1], marker="o", color=self.COLOR_MAP["pc"])

    def handle_pc(self, pc):
        plt.scatter(pc[:, 0], pc[:, 1], marker="o", color=self.COLOR_MAP["pc"])

    def handle_pc_3d(self, pc):
        plt.clf()
        ax = self.figure.gca(projection="3d")
        X, Y, Z = pc[:, 0], pc[:, 1], pc[:, 2]
        ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
        if self.title is not None:
            ax.set_title(self.title, fontsize=self.figsize[0] * 2)
        ax.scatter(X, Y, Z, marker="o", color=self.COLOR_MAP["gt_bbox"])

    def handle_gt_bbox(self, gt_bbox):
        gt_corners = np.array(coord.box2corners2d(gt_bbox))[:, :2]
        # so that plt.plot can form a closed rectangle
        gt_corners = np.concatenate([gt_corners, gt_corners[0:1, :]])

        plt.plot(gt_corners[:, 0], gt_corners[:, 1], color=self.COLOR_MAP["gt_bbox"], linestyle=":")

    def handle_3d_gt_bbox(self, gt_bbox, transform_mat):
        corners = np.array(coord.box2corners3d(gt_bbox))
        corners = coord.transform(transform_mat, corners)

        plt.plot(
            corners[:2, 0],
            corners[:2, 1],
            corners[2:4, 0],
            corners[2:4, 1],
            color=self.COLOR_MAP["gt_bbox"],
            linestyle=":",
        )
        plt.plot(
            corners[4:6, 0],
            corners[4:6, 1],
            corners[6:8, 0],
            corners[6:8, 1],
            color=self.COLOR_MAP["gt_bbox"],
            linestyle=":",
        )
        plt.plot(
            corners[1:3, 0],
            corners[1:3, 1],
            corners[5:7, 0],
            corners[5:7, 1],
            color=self.COLOR_MAP["gt_bbox"],
            linestyle=":",
        )
        plt.plot(
            corners[2:5:2, 0],
            corners[2:5:2, 1],
            corners[3:6:2, 0],
            corners[3:6:2, 1],
            color=self.COLOR_MAP["gt_bbox"],
            linestyle=":",
        )
        plt.plot(
            corners[0:4:3, 0],
            corners[0:4:3, 1],
            corners[4:8:3, 0],
            corners[4:8:3, 1],
            color=self.COLOR_MAP["gt_bbox"],
            linestyle=":",
        )
        plt.plot(
            corners[0:7:6, 0],
            corners[0:7:6, 1],
            corners[1:8:6, 0],
            corners[1:8:6, 1],
            color=self.COLOR_MAP["gt_bbox"],
            linestyle=":",
        )

    def handle_proposal_box(self, proposal_box):
        proposal_corners = np.array(coord.box2corners2d(proposal_box))[:, :2]
        proposal_corners = np.concatenate([proposal_corners, proposal_corners[0:1, :]])
        plt.plot(proposal_corners[:, 0], proposal_corners[:, 1], color=self.COLOR_MAP["proposal_bbox"], linestyle="-")
