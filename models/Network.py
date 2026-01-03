import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from utils.misc import initialize_weights
from torch_geometric.nn import GCNConv
import math
from models.pvtv2 import pvt_v2_b3,pvt_v2_b2,pvt_v2_b4,pvt_v2_b1
from thop import profile

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        return out

class ResBlock1D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock1D, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # x: [B, C, L]   C=通道数, L=序列长度(光谱/时间)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class SA(nn.Module):
    def __init__(self, in_dim):
        super(SA, self).__init__()

        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z,x,y):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(y).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(z).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = z + out.view(m_batchsize, C, height, width)
        return out

class SA2(nn.Module):
    def __init__(self, in_dim):
        super(SA2, self).__init__()

        self.chanel_in = in_dim
        self.query = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)



        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z,x,y):
        m_batchsize, C, height, width = x.size()
        query_c = self.query(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        key_c = self.key(y).view(m_batchsize, -1, width * height)
        energy_c = torch.bmm(query_c, key_c)
        attention_c = self.softmax(energy_c)

        query_s = x.view(m_batchsize, C, -1)
        key_s = y.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy_s = torch.bmm(query_s, key_s)
        energy_s = torch.max(energy_s, -1, keepdim=True)[0].expand_as(energy_s) - energy_s
        attention_s  = self.softmax(energy_s)
        attention = self.softmax(attention_c*attention_s)

        value = self.value(z).view(m_batchsize, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = z+out.view(m_batchsize, C, height, width)
        return out

class TGI(torch.nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, out_dim=128, num_nodes=3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2= nn.Linear(hidden_dim, out_dim)

        self.adj_matrix = nn.Parameter(torch.ones(num_nodes, num_nodes)).cuda()  # 全连接

        self.attention = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        self.conv = nn.Conv2d(in_channels=input_dim*3, out_channels=input_dim, kernel_size=1)
        self.att_linear = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
    def get_adjacency_matrix(self):
        adj = F.relu(self.adj_matrix)
        return adj / (adj.sum(dim=1, keepdim=True) + 1e-10)

    def forward(self, x1, x2, x3):
        origin = self.conv(torch.cat([x1,x2,x3],dim=1))
        x1 = F.avg_pool2d(x1, kernel_size=(64, 64)).squeeze(-1).squeeze(-1)
        x2 = F.avg_pool2d(x2, kernel_size=(64, 64)).squeeze(-1).squeeze(-1)
        x3 = F.avg_pool2d(x3, kernel_size=(64, 64)).squeeze(-1).squeeze(-1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x3 = x3.unsqueeze(1)
        x = torch.cat([x1, x2, x3], dim=1)
        identity = x
        adj_matrix = self.get_adjacency_matrix()
        x = self.linear1(x)
        x = torch.matmul(adj_matrix, x)
        x = F.relu(x) + identity
        x = self.linear2(x)
        attn_weights = self.attention(x)
        x = self.att_linear(x * attn_weights)
        x = x.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        new = x * origin
        return new

class SGI(torch.nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512,out_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.linear = nn.Linear(input_dim, out_dim)

    def grid_edge_index(self,x):
        B, C, height, width= x.size()
        idx = torch.arange(height * width).view(height, width)
        right = torch.stack([idx[:, :-1].flatten(), idx[:, 1:].flatten()], dim=0)
        down = torch.stack([idx[:-1, :].flatten(), idx[1:, :].flatten()], dim=0)
        edge_index = torch.cat([right, down, right[[1, 0]], down[[1, 0]]], dim=1)
        return edge_index.long()

    def forward(self, x):
        B,C,L,W = x.size()
        edge_index = self.grid_edge_index(x).cuda()
        x = x.view(B, C, L * W).permute(0, 2, 1)
        identity = x
        origin = (self.linear(x))
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x+identity, edge_index)
        x = x*origin
        _,_,c_o = x.size()
        x = x.permute(0, 2, 1).view(B, c_o, L , W)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, pretrained_path=r'.\pvt_v2_b1.pth'):
        super(Encoder, self).__init__()
        self.backbone = pvt_v2_b1()  # [64, 128, 320, 512]
        save_model = torch.load(pretrained_path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.Translayer0 = BasicConv2d(64, 128, 3, padding=1)
        self.Translayer1 = BasicConv2d(128, 128, 3, padding=1)
        self.Translayer2 = BasicConv2d(320, 128, 3, padding=1)
        self.Translayer3 = BasicConv2d(512, 128, 3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.conv = self._make_layer(ResBlock, 128 * 4, 512, 1, stride=1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x1, x2,x3):

        x1 = self.backbone(x1)
        x1_1 = self.Translayer0(x1[0])
        x1_2 = self.up1(self.Translayer1(x1[1]))
        x1_3 = self.up2(self.Translayer2(x1[2]))
        x1_4 = self.up3(self.Translayer3(x1[3]))
        x1_4 = self.conv(torch.cat([x1_1,x1_2,x1_3,x1_4],dim=1))

        x2 = self.backbone(x2)
        x2_1 = self.Translayer0(x2[0])
        x2_2 = self.up1(self.Translayer1(x2[1]))
        x2_3 = self.up2(self.Translayer2(x2[2]))
        x2_4 = self.up3(self.Translayer3(x2[3]))
        x2_4 = self.conv(torch.cat([x2_1, x2_2, x2_3, x2_4], dim=1))

        x3 = self.backbone(x3)
        x3_1 = self.Translayer0(x3[0])
        x3_2 = self.up1(self.Translayer1(x3[1]))
        x3_3 = self.up2(self.Translayer2(x3[2]))
        x3_4 = self.up3(self.Translayer3(x3[3]))
        x3_4 = self.conv(torch.cat([x3_1, x3_2, x3_3, x3_4], dim=1))




        return x1_4, x2_4, x3_4


class STG(nn.Module):
    def __init__(self,  num_classes=2):
        super(STG, self).__init__()
        self.Encoder = Encoder()

        self.TGI = TGI(input_dim=512,hidden_dim=512,out_dim=512)
        self.SGI = SGI(input_dim=512*3, hidden_dim=512*3, out_dim=512)
        self.stfusion = self._make_layer(ResBlock, 512 * 2, 512, 1, stride=1)
        self.conv2 = self._make_layer(ResBlock, 512 * 4, 128, 1, stride=1)

        self.CA = SA(512)
        self.intrnal = self._make_layer(ResBlock, 512* 4, 512, 1, stride=1)
        self.conv_global = self._make_layer(ResBlock, 512, 64, 1, stride=1)
        self.conv_ccd = self._make_layer(ResBlock, 512 , 64, 1, stride=1)

        self.classifier_global = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        self.classifier_ccd  = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

        # initialize_weights( self.backbone_res,self.backbone_non_res,self.change_diff,self.SA, self.change_res2,self.classifierCD,self.classifierNCD )

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2,x3):
        x_size = x1.size()

        x1,x2,x3 = self.Encoder(x1, x2,x3)

        x_n_1 = x1
        x_n_2 = x2
        x_n_3 = x3

        x_change_temporal = self.TGI(x_n_1,x_n_2,x_n_3)
        x_change_sptail = self.SGI(torch.cat([x_n_1, x_n_2, x_n_3],dim=1))
        g_change = self.stfusion(torch.cat([x_change_temporal,x_change_sptail],dim=1))

        d_1 = self.intrnal(torch.cat([x1,x2,torch.abs(x1 - x2), x1*x2],dim=1))
        d_2 = self.intrnal(torch.cat([x2,x3,torch.abs(x2 - x3), x2*x3],dim=1))

        d_1 = self.CA(d_1, d_1,g_change)
        d_2 = self.CA(d_2, d_1,g_change)

        global_change = self.classifier_global(self.conv_global(g_change))
        continuous_change1 = self.classifier_ccd(self.conv_ccd (d_1))
        continuous_change2 = self.classifier_ccd(self.conv_ccd (d_2))

        return F.interpolate(global_change, x_size[2:], mode='bilinear'),F.interpolate( continuous_change1, x_size[2:], mode='bilinear'),F.interpolate( continuous_change2, x_size[2:], mode='bilinear')


if __name__ == '__main__':
    x1 = torch.randn(1, 3, 256, 256).cuda()
    x2 = torch.randn(1, 3, 256, 256).cuda()
    model =  STG(num_classes=2).cuda()
    T1,T2,T3 = model(x1,x1,x1)
    print(T1.shape)
    print(T2.shape)
    print(T3.shape)


