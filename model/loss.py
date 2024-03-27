# -*- coding:utf8 -*-

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import bbox_iou, xyxy2xywh


def MSELoss( pred, target):
    return torch.pow(pred - target, 2)

def clip_by_tensor( t, t_min, t_max):
    t = t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result




def BCELoss( pred, target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


def get_ignore( x, y, h, w, targets, ref_scaled_anchors , in_h, in_w, noobj_mask,mode):
    # -----------------------------------------------------#
    #   计算一共有多少张图片
    # -----------------------------------------------------#
    bs = len(targets)
    # -----------------------------------------------------#
    #   生成网格，先验框中心，网格左上角
    # -----------------------------------------------------#
    grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
        int(bs * len(ref_scaled_anchors )), 1, 1).view(x.shape).type_as(x)
    grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
        int(bs * len(ref_scaled_anchors )), 1, 1).view(y.shape).type_as(x)

    # 生成先验框的宽高
    scaled_anchors_l = np.array(ref_scaled_anchors )
    anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
    anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)
    #anchor_w形状为[batch，len(anchor),64,64]每一行均为ref_scaled_anchors _l中的w
    anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
    anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
    # -------------------------------------------------------#
    #   计算调整后的先验框中心与宽高
    # -------------------------------------------------------#
    # pred_boxes_x = torch.unsqueeze(x + grid_x, -1)
    # pred_boxes_y = torch.unsqueeze(y + grid_y, -1)
    # pred_boxes_w = torch.unsqueeze(torch.exp(w) * anchor_w, -1)
    # pred_boxes_h = torch.unsqueeze(torch.exp(h) * anchor_h, -1)

    pred_boxes_x = torch.unsqueeze(x , -1)
    pred_boxes_y = torch.unsqueeze(y , -1)
    pred_boxes_w = torch.unsqueeze(torch.exp(w) * anchor_w, -1)
    pred_boxes_h = torch.unsqueeze(torch.exp(h) * anchor_h, -1)


    pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)

    for b in range(bs):
        # -------------------------------------------------------#
        #   将预测结果转换一个形式
        #   pred_boxes_for_ignore      num_anchors, 4
        # -------------------------------------------------------#
        pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)
        # -------------------------------------------------------#
        #   计算真实框，并把真实框转换成相对于特征层的大小
        #   gt_box      num_true_box, 4
        # -------------------------------------------------------#
        if len(targets[b]) > 0:
            batch_target = torch.zeros_like(targets[b])
            # -------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            # -------------------------------------------------------#
            if mode=="ref_image":
                batch_target[:, 0] = (targets[b][:, 0] + targets[b][:, 2]) / ((1024 / 64.0)*2)
                batch_target[:, 1] = (targets[b][:, 1] + targets[b][:, 3]) / ((1024 / 64.0)*2)
                batch_target[:, 2] = (targets[b][:, 2] - targets[b][:, 0]) / (1024 / 64.0)  # w
                batch_target[:, 3] = (targets[b][:, 3] - targets[b][:, 1]) / (1024 / 64.0)  # h
            if mode=="query_image":
                batch_target[:, 0] = (targets[b][:, 4] + targets[b][:, 6]) / ((512 / 64.0)*2) #x
                batch_target[:, 1] = (targets[b][:, 5] + targets[b][:, 7]) /  ((256 / 64.0)*2) #y
                batch_target[:, 2] = (targets[b][:, 6] - targets[b][:, 4]) / (512 / 64.0)  # w
                batch_target[:, 3] = (targets[b][:, 7] - targets[b][:, 5]) / (256 / 64.0)  # h


            # batch_target[:, [0, 2]] = targets[b][:, [0, 2]] /(1024/64.0)
            # batch_target[:, [1, 3]] = targets[b][:, [1, 3]]/(1024/64.0)

            # batch_target[:, 4] = targets[b][:, 4] / (1232/ 64.0)
            # batch_target[:, 5] = targets[b][:, 5] / (1232 / 64.0)



            batch_target = batch_target[:, :4].type_as(x)
            # -------------------------------------------------------#
            #   计算交并比
            #   anch_ious       num_true_box, num_anchors
            # -------------------------------------------------------#
            anch_ious = calculate_iou(batch_target, pred_boxes_for_ignore)
            # -------------------------------------------------------#
            #   每个先验框对应真实框的最大重合度
            #   anch_ious_max   num_anchors
            # -------------------------------------------------------#
            anch_ious_max, _ = torch.max(anch_ious, dim=0)
            anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
            noobj_mask[b][anch_ious_max > 0.5] = 0
    return noobj_mask, pred_boxes



def calculate_iou( _box_a, _box_b):
    # -----------------------------------------------------------#
    #   计算真实框的左上角和右下角
    # -----------------------------------------------------------#
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    # -----------------------------------------------------------#
    #   计算先验框获得的预测框的左上角和右下角
    # -----------------------------------------------------------#
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

    # -----------------------------------------------------------#
    #   将真实框和预测框都转化成左上角右下角的形式
    # -----------------------------------------------------------#
    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

    # -----------------------------------------------------------#
    #   A为真实框的数量，B为先验框的数量
    # -----------------------------------------------------------#
    A = box_a.size(0)
    B = box_b.size(0)

    # -----------------------------------------------------------#
    #   计算交的面积
    # -----------------------------------------------------------#
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]
    # -----------------------------------------------------------#
    #   计算预测框和真实框各自的面积
    # -----------------------------------------------------------#
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    # -----------------------------------------------------------#
    #   求IOU
    # -----------------------------------------------------------#
    union = area_a + area_b - inter
    return inter / union  # [A,B]




def adjust_learning_rate(args, optimizer, i_iter):
    
    lr = args.lr*((0.1)**(i_iter//10))
        
    print(("lr", lr))
    for param_idx, param in enumerate(optimizer.param_groups):
        param['lr'] = lr

# the shape of the target is (batch_size, anchor_count, 5, grid_wh, grid_wh)
def yolo_loss(predictions, gt_bboxes, ref_anchors_full, best_anchor_gi_gj, image_wh,data_mode):
    if data_mode == 'CVOGL_DroneAerial' or data_mode == 'CVOGL_SVI':
        # 获取预测张量的尺寸信息，其中predictions的形状为(batch_size, anchor_count, num_classes + 5, grid_h, grid_w)。
        # 根据预测张量的形状，batch_size表示批次大小，anchor_count表示锚框数量，num_classes表示目标类别数，grid_h和grid_w表示预测网格的高度和宽度。
        # 计算网格步幅(grid_stride)，即图像宽度(image_wh)除以预测网格的宽度(grid_w)。
        batch_size, grid_stride = predictions.shape[0], image_wh // predictions.shape[3]

        # 从best_anchor_gi_gj中获取最佳锚框索引(best_anchor)、网格坐标(gi)和(gj)。
        best_anchor, gi, gj = best_anchor_gi_gj[:, 0], best_anchor_gi_gj[:, 1], best_anchor_gi_gj[:, 2]

        # 将锚框(ref_anchors_full)按网格步幅(grid_stride)进行缩放，得到缩放后的锚框(ref_scaled_anchors )。
        ref_scaled_anchors  = ref_anchors_full / grid_stride
        mseloss = torch.nn.MSELoss(size_average=True)
        celoss_confidence = torch.nn.CrossEntropyLoss(size_average=True)
        # celoss_cls = torch.nn.CrossEntropyLoss(size_average=True)

        # 从预测张量(predictions)中选择最佳锚框对应的预测值(selected_predictions)，用于计算边界框损失。[12,9,5,64,64]
        selected_predictions = predictions[range(batch_size), best_anchor, :, gj, gi]

        # ---bbox loss---
        # 根据预测的边界框值(selected_predictions)，将其转换为中心点和宽高格式(pred_bboxes)。
        pred_bboxes = torch.zeros_like(gt_bboxes)
        # 网络预测的x，y并不在0-1范围内，使用simoid（）函数将偏移量变换到0-1之间。
        pred_bboxes[:, 0:2] = selected_predictions[:, 0:2].sigmoid()
        pred_bboxes[:, 2:4] = selected_predictions[:, 2:4]

        # 计算边界框的坐标损失，分别计算中心点的坐标损失(loss_x和loss_y)，以及宽度和高度的损失(loss_w和loss_h)。
        loss_x = mseloss(pred_bboxes[:, 0], gt_bboxes[:, 0])
        loss_y = mseloss(pred_bboxes[:, 1], gt_bboxes[:, 1])
        loss_w = mseloss(pred_bboxes[:, 2], gt_bboxes[:, 2])
        loss_h = mseloss(pred_bboxes[:, 3], gt_bboxes[:, 3])

        # 将边界框的坐标损失相加，得到边界框损失(loss_bbox)。
        loss_bbox = loss_x + loss_y + loss_w + loss_h

        # ---confidence loss---
        # 从预测张量中提取置信度预测值(pred_confidences)。
        pred_confidences = predictions[:, :, 4, :, :]

        # 根据最佳锚框的网格坐标(gi和gj)构造目标置信度(gt_confidences)，其中最佳锚框对应的位置为1，其他位置为0。
        gt_confidences = torch.zeros_like(pred_confidences)
        gt_confidences[range(batch_size), best_anchor, gj, gi] = 1

        # 将预测的置信度和目标置信度重塑成二维张量，并计算交叉熵损失(loss_confidence)。
        pred_confidences, gt_confidences = pred_confidences.reshape(batch_size, -1), \
            gt_confidences.reshape(batch_size, -1)
        loss_confidence = celoss_confidence(pred_confidences, gt_confidences.max(1)[1])

        return loss_bbox, loss_confidence


#target_coord:batch_size, 5
#ori_gt_bboxe表示gt框信息
#ref_anchors_full表示anchor信息
#image_wh模型读取的图像宽高信息（1024）
#grid_wh模型预测特征图的宽高
#data_mode表示数据集的类别信息
def  build_target(ori_gt_bboxes, ref_anchors_full, image_wh, grid_wh,data_mode):

    if data_mode=='CVOGL_DroneAerial' or data_mode=='CVOGL_SVI':
        # the default value of coord_dim is 5
        # ori_gt_bboxes.shape为【batch，gt框坐标】
        batch_size, coord_dim, grid_stride, anchor_count = ori_gt_bboxes.shape[0], ori_gt_bboxes.shape[
            1], image_wh // grid_wh, ref_anchors_full.shape[0]

        gt_bboxes = xyxy2xywh(ori_gt_bboxes)
        gt_bboxes = (gt_bboxes * grid_wh) / image_wh  # 将gt_bboxes的尺寸缩放到特征图对应的尺寸
        # 将锚框(ref_anchors_full)按照网格的步幅(grid_stride)进行缩放。
        ref_scaled_anchors  = ref_anchors_full / grid_stride

        # 提取ground truth边界框的中心点坐标(gxy)和宽高(gwh)，并将中心点坐标转换为网格坐标(gij)
        # gt_bboxes[:, 0:2]提取了gt_bboxes张量中所有行的第0列和第1列的数据
        gxy = gt_bboxes[:, 0:2]
        gwh = gt_bboxes[:, 2:4]
        gij = gxy.long()

        # get the best anchor for each target bbox
        # 创建临时的ground truth边界框张量(gt_bboxes_tmp)和缩放后的锚框张量(ref_scaled_anchors _tmp)，用于计算ground truth边界框与锚框之间的IoU。
        gt_bboxes_tmp, scaled_anchors_tmp = torch.zeros_like(gt_bboxes), torch.zeros((anchor_count, coord_dim),
                                                                                     device=gt_bboxes.device)
        gt_bboxes_tmp[:, 2:4] = gwh
        gt_bboxes_tmp = gt_bboxes_tmp.unsqueeze(1).repeat(1, anchor_count, 1).view(-1, coord_dim)  # [108,4]

        scaled_anchors_tmp[:, 2:4] = ref_scaled_anchors
        scaled_anchors_tmp = scaled_anchors_tmp.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, coord_dim)

        anchor_ious = bbox_iou(gt_bboxes_tmp, scaled_anchors_tmp).view(batch_size, -1)
        best_anchor = torch.argmax(anchor_ious, dim=1)  # 在anchor_ious张量的第1维度上寻找最大值，并返回最大值的索引。

        # 计算目标边界框的高度和宽度相对于最匹配的锚框的缩放比例(twh)，并应用对数变换。
        twh = torch.log(gwh / ref_scaled_anchors [best_anchor] + 1e-16)
        # print((gxy.dtype, gij.dtype, twh.dtype, gwh.dtype, scaled_anchors.dtype, 'inner'))
        # print((gxy.shape, gij.shape, twh.shape, gwh.shape), flush=True)
        # print(('gxy,gij,twh', gxy, gij, twh), flush=True)

        # 将ground truth边界框的网格坐标偏移(gxy - gij)和缩放比例(twh)拼接成一个张量，并返回结果。
        # 将最匹配的锚框索引(best_anchor)和网格坐标(gij)拼接成一个张量，并返回结果。


        return torch.cat((gxy - gij, twh), 1), torch.cat((best_anchor.unsqueeze(1), gij), 1),None  # [12,4]、[12,3]


#targets:标注框
def get_target( targets, ref_anchors,query_anchors, in_h, in_w,bbox_attrs):
    # -----------------------------------------------------#
    #   计算一共有多少张图片
    # -----------------------------------------------------#
    bs = len(targets)
    # -----------------------------------------------------#
    #   用于选取哪些先验框不包含物体
    # -----------------------------------------------------#
    # [bs,3,13,13]
    ref_noobj_mask = torch.ones(bs, len(ref_anchors), in_h, in_w, requires_grad=False)
    query_noobj_mask = torch.ones(bs, len(ref_anchors), in_h, in_w, requires_grad=False)
    # -----------------------------------------------------#
    #   让网络更加去关注小目标
    # -----------------------------------------------------#
    box_loss_scale = torch.zeros(bs, len(ref_anchors), in_h, in_w, requires_grad=False)
    # -----------------------------------------------------#
    #   batch_size, 3, 13, 13, 5 + num_classes
    # -----------------------------------------------------#
    y_true = torch.zeros(bs, len(ref_anchors), in_h, in_w, bbox_attrs, requires_grad=False)
    for b in range(bs):
        if len(targets[b]) == 0:
            continue
        batch_target = torch.zeros_like(targets[b])
        # -------------------------------------------------------#
        #   计算出正样本在特征层上的中心点,此时target的形状为【ref_img坐标，query_img坐标】
        # -------------------------------------------------------#
        batch_target[:, [0, 2]] = targets[b][:, [0, 2]]/(1024/64.0)
        batch_target[:, [1, 3]] = targets[b][:, [1, 3]]/(1024/64.0)

        batch_target[:, [4, 6]] = targets[b][:,4] / (512 / 64.0)
        batch_target[:, [5, 7]] = targets[b][:, 5]/ (256 / 64.0)

        batch_target = batch_target.cpu()

        # -------------------------------------------------------#
        #   将真实框转换一个形式
        # batch_target.size(0)表示每个图像上gt的数目
        #   num_true_box, 4
        # -------------------------------------------------------#
        # gt_box.shape=[num_true_box, 8],存储形式为：【xmin，ymin，xmax，ymax，qury_xmin，query_ymin，query-xmax，query_ymax】
        #batch_target.shape=[b,4]
        ref_gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))
        query_gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 6:8]), 1))
        # -------------------------------------------------------#
        #   将先验框转换一个形式
        #   9, 4
        # -------------------------------------------------------#
        ref_anchor_shapes = torch.FloatTensor(
            torch.cat((torch.zeros((len(ref_anchors), 2)), torch.FloatTensor(ref_anchors)), 1))
        query_anchor_shapes = torch.FloatTensor(
            torch.cat((torch.zeros((len(query_anchors), 2)), torch.FloatTensor(query_anchors)), 1))

        # -------------------------------------------------------#
        #   计算交并比
        #   self.calculate_iou(gt_box, anchor_shapes) = [num_true_box, 9]每一个真实框和9个先验框的重合情况
        #   best_ns:
        #   [每个真实框最大的重合度max_iou, 每一个真实框最重合的先验框的序号]
        # -------------------------------------------------------#
        ref_best_ns = torch.argmax(calculate_iou(ref_gt_box, ref_anchor_shapes), dim=-1)
        query_best_ns = torch.argmax(calculate_iou(query_gt_box, query_anchor_shapes), dim=-1)




        for t, (ref_best_n,query_best_n) in enumerate(zip(ref_best_ns,query_best_ns)):
            # ----------------------------------------#
            #   判断这个先验框是当前特征点的哪一个先验框
            # ----------------------------------------#
            r_k = ref_best_n  # 查找best_n在self.anchors_mask[l]中的位置
            q_k =query_best_n

            # ----------------------------------------#
            #   获得真实框属于哪个网格点
            # ----------------------------------------#
            r_i = torch.floor(batch_target[t, 0]).long()
            r_j = torch.floor(batch_target[t, 1]).long()

            q_i = torch.floor(batch_target[t, 0]).long()
            q_j = torch.floor(batch_target[t, 1]).long()


            # ----------------------------------------#
            #   取出真实框的种类
            # ----------------------------------------#
            c = batch_target[t, 4].long()

            # ----------------------------------------#
            #   noobj_mask代表无目标的特征点
            # 记录每张图里每个目标最匹配的anchor索引及目标对应的网格坐标
            # ----------------------------------------#
            ref_noobj_mask[b, r_k, r_j, r_i] = 0
            query_noobj_mask[b, q_k, q_j, q_i] = 0

            # ----------------------------------------#
            #   tx、ty代表中心调整参数的真实值
            # ----------------------------------------#

             # ----------------------------------------#
            #   tx、ty代表中心调整参数的真实值
            # ----------------------------------------#
            y_true[b, r_k, r_j, r_i, 0] = (batch_target[t, 0]+ batch_target[t, 2] )/2.0 #gt框中心点x坐标
            y_true[b, r_k, r_j, r_i, 1] = (batch_target[t, 1]+ batch_target[t, 3] )/2.0
            y_true[b, r_k, r_j, r_i, 2] = (batch_target[t, 2]- batch_target[t, 0])
            y_true[b, r_k, r_j, r_i, 3] = (batch_target[t, 3]- batch_target[t, 1])
            y_true[b, r_k, r_j, r_i, 4] = 1 #有无目标的置信度
            y_true[b, q_k, q_j, q_i, 5] =(batch_target[t, 4]+ batch_target[t, 6] )/2.0
            y_true[b, q_k, q_j, q_i, 6] = (batch_target[t, 5]+ batch_target[t, 7] )/2.0
            # print(batch_target[t, 6] ,query_anchors[query_best_n][0])
            y_true[b,  q_k, q_j, q_i, 7] = (batch_target[t, 6]- batch_target[t, 4])
            y_true[b,  q_k, q_j, q_i, 8] = (batch_target[t, 7]- batch_target[t, 5])
            y_true[b, q_k, q_j, q_i, 9] = 1  # 有无目标的置信度

            # ----------------------------------------#
            #   用于获得xywh的比例
            #   大目标loss权重小，小目标loss权重大
            # ----------------------------------------#
            # box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        # y_true代表gt框信息在特征图上的位置， y_true[b, k, j, i, 0]、 y_true[b, k, j, i, 1]、 y_true[b, k, j, i, 2]、 y_true[b, k, j, i, 3]分别代表中心点坐标、宽、高
        # noobj_mask中值为0的地方，代表特征图的该位置存在目标
        # box_loss_scale中值不为0的位置代表当前目标计算时的权重。
    return y_true, ref_noobj_mask,query_noobj_mask,




def RRdata_loss(pred_anchor,ori_gt_bbox,ref_anchors_full,query_anchors_full,img_size):
    #9代表：中心点坐标+宽高+置信度+查询目标图像中目标的位置信息
    #pred_anchor=[batch,len(anchor)*7,h,w]
    #   获得图片数量，特征层的高和宽
    #   13和13
    # --------------------------------#
    bs = pred_anchor.size(0)
    in_h = pred_anchor.size(2)
    in_w = pred_anchor.size(3)
    #计算原图到特征图的缩放比例
    ref_stride_h = 750.0/ in_h
    ref_stride_w = 750.0/ in_w

    query_stride_h = 1232.0 / in_h
    query_stride_w = 224.0 / in_w

    #将anchor缩放到特征图对应的尺寸
    # -------------------------------------------------#
    ref_scaled_anchors = [(a_w.cpu() / ref_stride_h, a_h.cpu() / ref_stride_w) for a_w, a_h in ref_anchors_full]
    query_scaled_anchors = [(a_w.cpu() / query_stride_h, a_h.cpu() / query_stride_w) for a_w, a_h in query_anchors_full]


    # bbox_attrs=5+ num_classes
    prediction = pred_anchor.view(bs, len(ref_anchors_full), 10, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
    # -----------------------------------------------#
    #   先验框的中心位置的调整参数
    # -----------------------------------------------#
    # x = torch.sigmoid(prediction[..., 0])
    # y = torch.sigmoid(prediction[..., 1])
    ref_x = prediction[..., 0]
    ref_y = prediction[..., 1]

    # ---------------------------------------------
    #   先验框的宽高调整参数
    # -----------------------------------------------#
    ref_w = prediction[..., 2]
    ref_h = prediction[..., 3]
    # -----------------------------------------------#
    #   获得置信度，是否有物体
    # -----------------------------------------------#
    ref_conf = torch.sigmoid(prediction[..., 4])
    # -----------------------------------------------#
    #   种类置信度
    # -----------------------------------------------#
    # x_query = torch.sigmoid(prediction[..., 5])
    # y_query = torch.sigmoid(prediction[..., 6])
    query_x = prediction[..., 5]
    query_y = prediction[..., 6]

    query_w = prediction[..., 7]
    query_h = prediction[..., 8]
    query_conf = torch.sigmoid(prediction[..., 9])
    # -----------------------------------------------#
    #   获得网络应该有的预测结果
    # -----------------------------------------------#
    #ori_gt_bbox.shape=【b，4】
    y_true, ref_noobj_mask, query_noobj_mask = get_target(ori_gt_bbox, ref_scaled_anchors,query_scaled_anchors, in_h, in_w,bbox_attrs=10)

    # ---------------------------------------------------------------#
    #   将预测结果进行解码，判断预测结果和真实值的重合程度
    #   如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点
    #   作为负样本不合适
    # ----------------------------------------------------------------#
    ref_noobj_mask, pred_boxes = get_ignore(ref_x, ref_y, ref_h, ref_w,ori_gt_bbox, ref_scaled_anchors , in_h, in_w, ref_noobj_mask,mode="ref_img")
    query_noobj_mask, pred_boxes = get_ignore( query_x, query_y, query_h, query_w, ori_gt_bbox,query_scaled_anchors, in_h, in_w, query_noobj_mask,mode="query_img")

    y_true = y_true.type_as(ref_x)
    ref_noobj_mask = ref_noobj_mask.type_as(ref_x)
    query_noobj_mask = query_noobj_mask.type_as(query_x)

    # box_loss_scale = box_loss_scale.type_as(x)
    # --------------------------------------------------------------------------#
    #   box_loss_scale是真实框宽高的乘积，宽高均在0-1之间，因此乘积也在0-1之间。
    #   2-宽高的乘积代表真实框越大，比重越小，小框的比重更大。
    # --------------------------------------------------------------------------#
    # box_loss_scale = 2 - box_loss_scale

    loss = 0
    ref_obj_mask = y_true[..., 4] == 1
    query_obj_mask = y_true[..., 9] == 1
    n = torch.sum(ref_obj_mask)


    if n != 0:
        # -----------------------------------------------------------#
        #   计算中心偏移情况的loss，使用BCELoss效果好一些
        # -----------------------------------------------------------#
        loss_x = torch.mean(MSELoss(ref_x[ref_obj_mask], y_true[..., 0][ref_obj_mask]))
        loss_y = torch.mean(MSELoss(ref_y[ref_obj_mask], y_true[..., 1][ref_obj_mask]))
        # -----------------------------------------------------------#
        #   计算宽高调整值的loss
        # -----------------------------------------------------------#
        loss_w = torch.mean(MSELoss(ref_w[ref_obj_mask], y_true[..., 2][ref_obj_mask]) )
        loss_h = torch.mean(MSELoss(ref_h[ref_obj_mask], y_true[..., 3][ref_obj_mask]))
        ref_loss_loc = (loss_x + loss_y + loss_h + loss_w)


        loss_x_query = torch.mean(MSELoss(query_x[query_obj_mask], y_true[..., 5][query_obj_mask]))
        loss_y_query = torch.mean(MSELoss(query_y[query_obj_mask] , y_true[..., 6][query_obj_mask]))
        #   计算宽高调整值的loss
        # -----------------------------------------------------------#
        loss_w_query = torch.mean(MSELoss(query_w[query_obj_mask], y_true[..., 7][query_obj_mask]))
        loss_h_query = torch.mean(MSELoss(query_h[query_obj_mask], y_true[..., 8][query_obj_mask]))



        query_loss_loc = (loss_x_query + loss_y_query + loss_w_query + loss_h_query)

    ref_loss_conf = torch.mean(BCELoss(ref_conf, ref_obj_mask.type_as(ref_conf))[ref_noobj_mask.bool() | ref_obj_mask])*ref_loss_loc.item()
    query_loss_conf = torch.mean(BCELoss(query_conf, query_obj_mask.type_as(query_conf))[ref_noobj_mask.bool() | ref_obj_mask]) * query_loss_loc.item()



    loss += ref_loss_conf+query_loss_conf+ref_loss_loc+query_loss_loc
    # if n != 0:
    #     print(loss_loc * self.box_ratio, loss_cls * self.cls_ratio, loss_conf * self.balance[l] * self.obj_ratio)
    return loss












