# -*- coding:utf8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .darknet import *
import torchvision.models as models

class MyResnet(nn.Module):
    def __init__(self):
        super(MyResnet, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.avgpool = nn.Sequential()
        self.base_model.fc = nn.Sequential()

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        #print(('x1', x.shape), flush=True)
        x = self.base_model.layer2(x)
        #print(('x2', x.shape), flush=True)
        x = self.base_model.layer3(x)
        #print(('x3', x.shape), flush=True)
        x = self.base_model.layer4(x)
        #print(('x4', x.shape), flush=True)
        return x

class CrossViewFusionModule(nn.Module):
    def __init__(self):
        super(CrossViewFusionModule, self).__init__()

    # normlized global_query:B, D
    # normlized value: B, D, H, W
    def forward(self, global_query, value):
        global_query = F.normalize(global_query, p=2, dim=-1)
        value = F.normalize(value, p=2, dim=1)

        B, D, W, H = value.shape
        new_value = value.permute(0, 2, 3, 1).view(B, W*H, D)
        score = torch.bmm(global_query.view(B, 1, D), new_value.transpose(1,2))
        score = score.view(B, W*H)
        with torch.no_grad():
            score_np = score.clone().detach().cpu().numpy()
            max_score, min_score = score_np.max(axis=1), score_np.min(axis=1)
        
        attn = Variable(torch.zeros(B, H*W).cuda())
        for ii in range(B):
            attn[ii, :] = (score[ii] - min_score[ii]) / (max_score[ii] - min_score[ii])
        
        attn = attn.view(B, 1, W, H)
        context = attn * value
        return context, attn

class DetGeo(nn.Module):
    def __init__(self, emb_size=512, leaky=True,data_name="RRdata"):
        super(DetGeo, self).__init__()
        ## Visual model
        self.query_resnet = MyResnet()
        self.data_name=data_name
        
        self.reference_darknet = Darknet(config_path='./model/yolov3_rs.cfg')
        self.reference_darknet.load_weights('./saved_models/yolov3.weights')
        
        use_instnorm=False

        self.combine_clickptns_conv = ConvBatchNormReLU(4, 3, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.crossview_fusionmodule = CrossViewFusionModule()
		
        self.query_visudim = 512 
        self.reference_visudim = 512

        self.query_mapping_visu = ConvBatchNormReLU(self.query_visudim, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.reference_mapping_visu = ConvBatchNormReLU(self.reference_visudim, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)

        ## output head
        self.fcn_out = torch.nn.Sequential(
                ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm),
                nn.Conv2d(emb_size//2, 9*5, kernel_size=1))

        ## output head,用于RRdataset
        #7代表：中心点坐标、宽高、置信度、查询图像中目标框中心点坐标
        self.fcn_out2 = torch.nn.Sequential(
            ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm),
            nn.Conv2d(emb_size // 2, 9 * 10, kernel_size=1))



    def forward(self, query_imgs, reference_imgs, mat_clickptns):
        if self.data_name == 'CVOGL_DroneAerial':
            # mat_clickptns.shape=[6,256,256]
            mat_clickptns = mat_clickptns.unsqueeze(1)  # mat_clickptns.shape=[6,1,256,256]
            # ,query_imgs.shape= [6, 3, 256, 256]
            query_imgs = self.combine_clickptns_conv(
                torch.cat((query_imgs, mat_clickptns), dim=1))  # ,query_imgs.shape= [6, 3, 256, 256]

            query_fvisu = self.query_resnet(query_imgs)  # query_fvisu.shape= [6, 512, 8, 8]
            # len(reference_raw_fvisu)=3,reference_raw_fvisu三个数组存放的是三个输出通道的特征图[6, 1024, 32, 32]、[6, 512, 64, 64]、[6, 256, 128, 128]
            reference_raw_fvisu = self.reference_darknet(reference_imgs)
            # reference_raw_fvisu[1].shape=[6, 512, 64, 64]
            reference_fvisu = reference_raw_fvisu[1]  # reference_raw_fvisu[1].shape=[6, 512, 64, 64]

            query_fvisu = self.query_mapping_visu(query_fvisu)  # query_fvisu.shape= [6, 512, 8, 8]

            reference_fvisu = self.reference_mapping_visu(reference_fvisu)

            B, D, Hquery, Wquery = query_fvisu.shape
            B, D, Hreference, Wreference = reference_fvisu.shape

            # cross-view fusion
            query_gvisu = torch.mean(query_fvisu.view(B, D, Hquery * Wquery), dim=2, keepdims=False).view(B, D)
            # query_gvisu=[6,512],attn_score.shape=[6,1,64,64]
            fused_features, attn_score = self.crossview_fusionmodule(query_gvisu, reference_fvisu)
            # fused_features.shape=[6,512,64,64],attn_score.shape=[6,64,64]
            attn_score = attn_score.squeeze(1)
            # outbox.shape=[6,45,64,64]
            outbox = self.fcn_out(fused_features)


        """____________________________________________RRdataset______________________________________________________________________"""
        if self.data_name == 'RRdata':
            # mat_clickptns.shape=[6,256,256]
            mat_clickptns = mat_clickptns.unsqueeze(1)  # mat_clickptns.shape=[6,1,256,256]
            # ,query_imgs.shape= [6, 3, 256, 256]
            query_imgs = self.combine_clickptns_conv(
                torch.cat((query_imgs, mat_clickptns), dim=1))  # ,query_imgs.shape= [6, 3, 256, 256]

            query_fvisu = self.query_resnet(query_imgs)  # query_fvisu.shape= [6, 512, 8, 8]
            

            # len(reference_raw_fvisu)=3,reference_raw_fvisu三个数组存放的是三个输出通道的特征图[6, 1024, 32, 32]、[6, 512, 64, 64]、[6, 256, 128, 128]
            reference_raw_fvisu = self.reference_darknet(reference_imgs)
            # reference_raw_fvisu[1].shape=[6, 512, 64, 64]
            reference_fvisu = reference_raw_fvisu[1]  # reference_raw_fvisu[1].shape=[6, 512, 64, 64]

            query_fvisu = self.query_mapping_visu(query_fvisu)  # query_fvisu.shape= [6, 512, 8, 8]

            reference_fvisu = self.reference_mapping_visu(reference_fvisu)

            B, D, Hquery, Wquery = query_fvisu.shape
            B, D, Hreference, Wreference = reference_fvisu.shape

            # cross-view fusion
            query_gvisu = torch.mean(query_fvisu.view(B, D, Hquery * Wquery), dim=2, keepdims=False).view(B, D)
            # query_gvisu=[6,512],attn_score.shape=[6,1,64,64]
            fused_features, attn_score = self.crossview_fusionmodule(query_gvisu, reference_fvisu)
            # fused_features.shape=[6,512,64,64],attn_score.shape=[6,64,64]
            attn_score = attn_score.squeeze(1)
            # outbox.shape=[6,45,64,64]
            outbox = self.fcn_out2(fused_features)

        return outbox, attn_score