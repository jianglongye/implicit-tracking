import torch
from torch import nn


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class ResnetPointnet(nn.Module):
    """PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    """

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c


class ResnetSDFDecoder(nn.Module):
    def __init__(self, c_dim=512, dim=3, hidden_dim=512):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, hidden_dim)
        self.block_0 = ResnetBlockFC(hidden_dim)
        self.block_1 = ResnetBlockFC(hidden_dim)
        self.block_2 = ResnetBlockFC(hidden_dim)

        self.block_3 = ResnetBlockFC(hidden_dim + c_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(hidden_dim)
        self.block_5 = ResnetBlockFC(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

        self.actvn = nn.ReLU()

    def forward(self, p, c):
        # p.shape: B x N x D
        # c.shape: B x D
        B, N, D = p.shape

        net = self.fc_pos(p)
        net = self.block_0(net)
        net = self.block_1(net)
        net = self.block_2(net)

        c = c.unsqueeze(1).expand(-1, N, -1)
        net = torch.cat([net, c], dim=2)

        net = self.block_3(net)
        net = self.block_4(net)
        net = self.block_5(net)

        sdf = self.fc_out(self.actvn(net))
        sdf = sdf.squeeze(2)

        return sdf  # sdf.shape: B x N


class PointSDFModel(nn.Module):
    def __init__(self, code_dim, hidden_dim, point_feat_dims, decoder_dims, use_res_decoder=False):
        super().__init__()
        self.encoder = ResnetPointnet(c_dim=code_dim, dim=3, hidden_dim=hidden_dim)

        self.use_res_decoder = use_res_decoder
        if use_res_decoder:
            self.decoder = ResnetSDFDecoder(c_dim=code_dim, dim=3, hidden_dim=hidden_dim)
        else:
            point_feat_layers = []
            for i in range(1, len(point_feat_dims)):
                point_feat_layers.append(nn.Conv1d(point_feat_dims[i - 1], point_feat_dims[i], 1))
                point_feat_layers.append(nn.ReLU())
            self.point_feature = nn.Sequential(*point_feat_layers)

            decoder_layers = []
            for i in range(1, len(decoder_dims)):
                decoder_layers.append(nn.Conv1d(decoder_dims[i - 1], decoder_dims[i], 1))
                if i < len(decoder_dims) - 1:
                    decoder_layers.append(nn.ReLU())
            self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, points, surface):
        glo_feat = self.encoder(surface)

        if self.use_res_decoder:
            pred_sdf = self.decoder(points.permute(0, 2, 1), glo_feat)
        else:
            point_feat = self.point_feature(points)
            pred_sdf = self.decoder(
                torch.cat((glo_feat.unsqueeze(-1).expand(-1, -1, point_feat.shape[-1]), point_feat), dim=1)
            )
        return glo_feat, pred_sdf

    def encode(self, surface):
        glo_feat = self.encoder(surface)
        return glo_feat

    def decode(self, points, code):
        if self.use_res_decoder:
            pre_sdf = self.decoder(points.permute(0, 2, 1), code)
        else:
            point_feat = self.point_feature(points)
            code = code.unsqueeze(-1).expand(-1, -1, points.shape[-1])
            pre_sdf = self.decoder(torch.cat((code, point_feat), dim=1))
        return pre_sdf
