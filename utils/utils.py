# -*- coding:utf8 -*-

import random

import cv2
import numpy as np
import torch
from torchvision.ops import nms

import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def xyxy2xywh(x):  # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    #y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y = torch.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

def xywh2xyxy(x):  # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    #y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y = torch.zeros_like(x)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y
    
def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # print(box1, box1.shape)
    # print(box2, box2.shape)
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def multiclass_metrics(pred, gt):
  """
  check precision and recall for predictions.
  Output: overall = {precision, recall, f1}
  """
  eps=1e-6
  overall = {'precision': -1, 'recall': -1, 'f1': -1}
  NP, NR, NC = 0, 0, 0  # num of pred, num of recall, num of correct
  for ii in range(pred.shape[0]):
    pred_ind = np.array(pred[ii]>0.5, dtype=int)
    gt_ind = np.array(gt[ii]>0.5, dtype=int)
    inter = pred_ind * gt_ind
    # add to overall
    NC += np.sum(inter)
    NP += np.sum(pred_ind)
    NR += np.sum(gt_ind)
  if NP > 0:
    overall['precision'] = float(NC)/NP
  if NR > 0:
    overall['recall'] = float(NC)/NR
  if NP > 0 and NR > 0:
    overall['f1'] = 2*overall['precision']*overall['recall']/(overall['precision']+overall['recall']+eps)
  return overall

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

#target_gj : target_bbox[:, :, :, x, :]
#target_gi : target_bbox[:, :, :, :, x]
def eval_iou_acc(pred_anchor, target_bbox, anchors_full, target_gi, target_gj, image_wh, iou_threshold_list=[0.5]):
    #print(pred_anchor)

    batch_size, grid_stride = target_bbox.shape[0], image_wh // pred_anchor.shape[3]
    #batch_size, anchor_count, xywh+confidence, grid_height, grid_width
    assert(len(pred_anchor.shape) == 5)
    assert(pred_anchor.shape[3] == pred_anchor.shape[4])
    
    ## eval: convert center+offset to box prediction
    ## calculate at rescaled image during validation for speed-up
    #张量的形状：[12，9，5，64，64]，这里相当于取出每个特征图的第5层通道数据即置信度
    pred_confidence = pred_anchor[:,:,4,:,:]
    scaled_anchors = anchors_full / grid_stride
    
    pred_gi, pred_gj = torch.zeros_like(target_gi), torch.zeros_like(target_gj)
    pred_bbox = torch.zeros_like(target_bbox)
    for batch_idx in range(batch_size):
        #寻找最大的置信度及其所在位置
        best_n, gj, gi = torch.where(pred_confidence[batch_idx].max() == pred_confidence[batch_idx])
        best_n, gj, gi = best_n[0], gj[0], gi[0]
        pred_gj[batch_idx], pred_gi[batch_idx] = gj, gi
        #print((best_n, gi, gj))

        #pred_anchor[batch_idx, best_n, 0, gj, gi]分别表示：当前的批次，最佳anchor的索引值，特征图中第0个维度、最佳置信度在特征图中的位置
        pred_bbox[batch_idx, 0] = pred_anchor[batch_idx, best_n, 0, gj, gi].sigmoid() + gi
        pred_bbox[batch_idx, 1] = pred_anchor[batch_idx, best_n, 1, gj, gi].sigmoid() + gj
        pred_bbox[batch_idx, 2] = torch.exp(pred_anchor[batch_idx, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
        pred_bbox[batch_idx, 3] = torch.exp(pred_anchor[batch_idx, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
    pred_bbox = pred_bbox * grid_stride
    pred_bbox = xywh2xyxy(pred_bbox)
    
    ## box iou
    iou = bbox_iou(pred_bbox, target_bbox, x1y1x2y2=True)
    each_acc50 = iou>0.5
    accu_list, each_acc_list=list(), list()
    for threshold in iou_threshold_list:
        each_acc = iou>threshold
        accu = torch.sum(each_acc)/batch_size
        accu_list.append(accu)
        each_acc_list.append(each_acc)
    accu_center = torch.sum((target_gi == pred_gi) * (target_gj == pred_gj))/batch_size
    iou = torch.sum(iou)/batch_size

    return accu_list, accu_center, iou, each_acc_list, pred_bbox, target_bbox


#input_shape=1024
def decode_box(input,input_shape,anchors_full):
    num_classes = 1
    bbox_attrs=5
    # -----------------------------------------------#
    #   输入的input一共有三个，他们的shape分别是
    #   batch_size, 255, 13, 13
    #   batch_size, 255, 26, 26
    #   batch_size, 255, 52, 52
    # -----------------------------------------------#
    batch_size = input.size(0)
    input_height = input.size(3)
    input_width = input.size(4)

    # -----------------------------------------------#
    #   输入为416x416时
    #   stride_h = stride_w = 32、16、8
    # -----------------------------------------------#
    stride_h = input_shape / input_height
    stride_w = input_shape/ input_width
    # -------------------------------------------------#
    #   此时获得的scaled_anchors大小是相对于特征层的
    # -------------------------------------------------#
    scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                      anchors_full]

    # -----------------------------------------------#
    #   输入的input一共有三个，他们的shape分别是
    #   batch_size, 3, 13, 13, 85
    #   batch_size, 3, 26, 26, 85
    #   batch_size, 3, 52, 52, 85
    # -----------------------------------------------#
    prediction = input.view(batch_size, len(anchors_full),
                            bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

    # -----------------------------------------------#
    #   先验框的中心位置的调整参数
    # -----------------------------------------------#
    x = prediction[..., 0]
    y = prediction[..., 1]
    # -----------------------------------------------#
    #   先验框的宽高调整参数
    # -----------------------------------------------#
    w = prediction[..., 2]
    h = prediction[..., 3]
    # -----------------------------------------------#
    #   获得置信度，是否有物体
    # -----------------------------------------------#
    conf = torch.sigmoid(prediction[..., 4])
    # -----------------------------------------------#
    #   种类置信度
    # -----------------------------------------------#
    # query_x_center = torch.sigmoid(prediction[..., 5])
    # query_y_center = torch.sigmoid(prediction[..., 6])

    FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

    # ----------------------------------------------------------#
    #   生成网格，先验框中心，网格左上角
    #   batch_size,3,13,13
    # ----------------------------------------------------------#
    grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
        batch_size * len(anchors_full), 1, 1).view(x.shape).type(FloatTensor)
    grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
        batch_size * len(anchors_full), 1, 1).view(y.shape).type(FloatTensor)

    # ----------------------------------------------------------#
    #   按照网格格式生成先验框的宽高
    #   batch_size,3,13,13
    # ----------------------------------------------------------#
    anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
    anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
    anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
    anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)



    query_grid_x= torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
        batch_size * len(anchors_full), 1, 1).view(x.shape).type(FloatTensor)

    query_grid_y= torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
        batch_size * len(anchors_full), 1, 1).view(y.shape).type(FloatTensor)


    # ----------------------------------------------------------#
    #   利用预测结果对先验框进行调整
    #   首先调整先验框的中心，从先验框中心向右下角偏移
    #   再调整先验框的宽高。
    # ----------------------------------------------------------#
    pred_boxes = FloatTensor(prediction[..., :4].shape)
    pred_boxes[..., 0] = x.data
    pred_boxes[..., 1] = y.data
    pred_boxes[..., 2] = w.data
    pred_boxes[..., 3] = h.data

    # query_boxes = FloatTensor(prediction[..., :2].shape)
    # query_boxes[..., 0] = query_x_center.data + query_grid_x
    # query_boxes[..., 1] = query_y_center.data + query_grid_y


    # ----------------------------------------------------------#
    #   将输出结果归一化成小数的形式
    # ----------------------------------------------------------#
    _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
    # query__scale = torch.Tensor([input_width, input_height]).type(FloatTensor)
    output = torch.cat((pred_boxes.view(batch_size, -1, 4) ,
                        conf.view(batch_size, -1, 1)), -1)
    return output.data


def yolo_correct_boxes( box_xy, box_wh, input_shape, image_shape, letterbox_image):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            #-----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            #-----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale
        #将坐标转化为左上角和右下角的形式
        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        #将boxes 数组的每个元素与 np.concatenate([image_shape, image_shape], axis=-1) 中的相应元素进行逐元素乘法。
        #ref：64->1024,64->1024
        #query:64->512,64->256
        scaled=input_shape/image_shape*1.0
        boxes *= np.concatenate([scaled, scaled], axis=-1)
        return boxes




def non_max_suppression( prediction, conf_thres=0.5, nms_thres=0.4):
    num_classes=1
    # ----------------------------------------------------------#
    #   将预测结果的格式转换成左上角右下角的格式。
    #   prediction  [batch_size, num_anchors, 85]
    # ----------------------------------------------------------#
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2 #x
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2 #y
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2 #w
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2 #h

    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))] #创建与prediction 长度相同的列表
    for i, image_pred in enumerate(prediction):
        # ----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        # ----------------------------------------------------------#
        #返回最大值及其对应的索引，这里class_conf代表最大值，class_pred代表最大值的索引
        # class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        # ----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        # ----------------------------------------------------------#
        #image_pred=【x1，y1，x2，y2，conf，类别（class）】
        conf_mask = (image_pred[:, 4]  >= conf_thres).squeeze()

        # ----------------------------------------------------------#
        #   根据置信度进行预测结果的筛选
        # ----------------------------------------------------------#
        image_pred = image_pred[conf_mask]
        # class_conf = class_conf[conf_mask]
        # class_pred = class_pred[conf_mask]
        if not image_pred.size(0):
            continue
        # -------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x, y, w, h, obj_conf, class_conf, class_pred
        # -------------------------------------------------------------------------#
        detections = (image_pred[:, :10])

        # ------------------------------------------#
        #   获得预测结果中包含的所有种类
        # ------------------------------------------#
        # unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            # unique_labels = unique_labels.cuda()
            detections = detections.cuda()

            # ------------------------------------------#
            #   获得某一类得分筛选后全部的预测结果
            # ------------------------------------------#
            #筛选出类别标签等于 c 的行
            detections_class = detections

            # ------------------------------------------#
            #   使用官方自带的非极大抑制会速度更快一些！
            # ------------------------------------------#
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4],
                nms_thres
            )
            max_detections = detections_class[keep]

            # Add max detections to outputs
            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i] = output[i].cpu().numpy()
            box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
            #网络读取图像input_shape的尺寸，image_shape图像的原始尺寸
            #将预测坐标点从64×64的特征图映射到输入尺寸上，ref输入到网络的图像尺寸为1024,1024]，query输入到网络的尺寸为：[512, 256]
            output[i][:, :4] = yolo_correct_boxes(box_xy, box_wh, input_shape=[1024,1024], image_shape=[64,64], letterbox_image=False)
            output[i][:, 5:9] = yolo_correct_boxes(box_xy, box_wh, input_shape=[512, 256], image_shape=[64, 64],letterbox_image=False)

    return output



