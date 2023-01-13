from pytorch3d.ops import knn_points
from torch.nn import functional as F


def chamfer_distance_loss(ori_x, ori_y, direction="y-x", loss_type="l1", smooth_l1_beta=0.03):
    loss = 0
    if direction == "x-y" or direction == "double":
        x_nn = knn_points(ori_x, ori_y)
        paired_y = ori_y[:, x_nn.idx[0, :, 0]]
        x = ori_x
        if loss_type == "l1":
            loss += F.l1_loss(paired_y, x)
        elif loss_type == "l2":
            loss += F.mse_loss(paired_y, x)
        elif loss_type == "smooth_l1":
            loss += F.smooth_l1_loss(paired_y, x, beta=smooth_l1_beta)

    if direction == "y-x" or direction == "double":
        y_nn = knn_points(ori_y, ori_x)
        paired_x = ori_x[:, y_nn.idx[0, :, 0]]
        y = ori_y

        if loss_type == "l1":
            loss += F.l1_loss(paired_x, y)
        elif loss_type == "l2":
            loss += F.mse_loss(paired_x, y)
        elif loss_type == "smooth_l1":
            loss += F.smooth_l1_loss(paired_x, y, beta=smooth_l1_beta)

    return loss
