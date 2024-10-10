import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_

from layers import ConvLayer

# The inertial encoder for raw imu data
class InertialEncoder(nn.Module):
    def __init__(self, par):
        super(InertialEncoder, self).__init__()

        imu_h0_len = par.imu_hidden_size//2 # length for the first layer
        
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, imu_h0_len, kernel_size=3, padding=1),
            nn.BatchNorm1d(imu_h0_len),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(par.dropout),
            nn.Conv1d(imu_h0_len, par.imu_hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(par.imu_hidden_size),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(par.dropout),
            nn.Conv1d(par.imu_hidden_size, par.imu_hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(par.imu_hidden_size),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(par.dropout))
        len_f = par.imu_per_image + 1 + par.imu_prev
        #len_f = (len_f - 1) // 2 // 2 + 1
        self.proj = nn.Linear(par.imu_hidden_size * 1 * len_f, par.imu_f_len)

    def forward(self, x):
        # x: (N, seq_len, 11, 6)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))    # x: (N x seq_len, 11, 6)

        x = self.encoder_conv(x.permute(0, 2, 1))    # x: (N x seq_len, 64, 11)
        out = self.proj(x.view(x.shape[0], -1))      # out: (N x seq_len, 256)
        return out.view(batch_size, seq_len, -1)


# The fusion module
class FusionModule(nn.Module):
    def __init__(self, par, temp=None):
        super(FusionModule, self).__init__()
        self.f_len = par.imu_f_len + par.visual_f_len

    def forward(self, v, i):
        return torch.cat((v, i), -1)

# The pose estimation network
class PoseMLP(nn.Module):
    def __init__(self, par):
        super(PoseMLP, self).__init__()

        # decoder mlp network
        f_len = par.visual_f_len + par.imu_f_len
        
        self.mlp = nn.Sequential(
                        nn.Linear(f_len, par.mlp_hidden_size),
                        nn.LeakyReLU(0.1, inplace=True),
                        nn.Linear(par.mlp_hidden_size, par.mlp_hidden_size))

        self.fuse = FusionModule(par)

        # The output networks
        self.mlp_drop_out = nn.Dropout(par.mlp_dropout_out)
        self.regressor = nn.Sequential(
            nn.Linear(par.mlp_hidden_size, par.mlp_hidden_size),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(par.mlp_hidden_size, 6))

    def forward(self, visual_f, imu_f, prev=None):

        # if prev is not None:
        #     prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())

        batch_size = visual_f.shape[0]
        seq_len = visual_f.shape[1]
        
        fused = self.fuse(visual_f, imu_f)

        out = self.mlp(fused)
        out = self.mlp_drop_out(out)
        pose = self.regressor(out)
        
        hc = None
        return pose, hc, out

class VisualEncoderLatency(nn.Module):
    def __init__(self, par):
        super(VisualEncoderLatency, self).__init__()
        # CNN
        self.par = par
        self.batchNorm = par.batch_norm
        # Visual Encoder Backbone from NASVIO
        self.conv1 = ConvLayer(6, 8, kernel_size=5, stride=2, use_norm=self.batchNorm, norm_func='BN', act_func='lrelu')
        self.conv2 = ConvLayer(8, 16, kernel_size=3, stride=2, use_norm=self.batchNorm, norm_func='BN', act_func='lrelu')
        self.conv3 = ConvLayer(16, 64, kernel_size=5, stride=2, use_norm=self.batchNorm, norm_func='BN', act_func='lrelu')
        self.conv4 = ConvLayer(64, 64, kernel_size=3, stride=2, use_norm=self.batchNorm, norm_func='BN', act_func='lrelu')
        self.conv4_1 = ConvLayer(64, 64, kernel_size=3, stride=1, use_norm=self.batchNorm, norm_func='BN', act_func='lrelu')
        self.conv5 = ConvLayer(64, 192, kernel_size=3, stride=2, use_norm=self.batchNorm, norm_func='BN', act_func='lrelu')
        self.conv5_1 = ConvLayer(192, 64, kernel_size=3, stride=1, use_norm=self.batchNorm, norm_func='BN', act_func='lrelu')
        self.conv6 = ConvLayer(64, par.visual_f_len, kernel_size=3, stride=2, use_norm=self.batchNorm, norm_func='BN', act_func='lrelu')
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.visual_head = nn.Linear(par.visual_f_len, par.visual_f_len)

    def forward(self, x, batch_size, seq_len):
        x = self.encode_image(x)
        x = self.avgpool(x).squeeze()
        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, fv)
        v_f = self.visual_head(x)  # (batch, seq_len, 256)
        return v_f

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

class ULVIO(nn.Module):
    def __init__(self, par):
        super(ULVIO, self).__init__()

        self.visual_encoder = VisualEncoderLatency(par)
        self.inertial_encoder = InertialEncoder(par)
        self.pose_net = PoseMLP(par)
        self.par = par

    def forward(self, t_x, t_i, prev=None):
        v_f, imu = self.encoder_forward(t_x, t_i)
        batch_size = v_f.shape[0]
        seq_len = v_f.shape[1]

        pose_list = []
        for i in range(seq_len):
            pose, prev, _ = self.pose_net(v_f[:, i, :].unsqueeze(1), imu[:, i, :].unsqueeze(1), prev)
            pose_list.append(pose)

        poses = torch.cat(pose_list, dim=1)
        return poses, prev

    def encoder_forward(self, v, imu):
        # x: (batch, seq_len, channel, width, height)
        # stack_image

        v = torch.cat((v[:, :-1], v[:, 1:]), dim=2)
        batch_size = v.size(0)
        seq_len = v.size(1)

        # CNN
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        v_f = self.visual_encoder(v, batch_size, seq_len)

        ll = 11 + self.par.imu_prev
        imu = torch.cat([imu[:, i * 10:i * 10 + ll, :].unsqueeze(1) for i in range(seq_len)], dim=1)
        imu = self.inertial_encoder(imu)

        return v_f, imu