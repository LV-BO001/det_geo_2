# -*- coding: utf8 -*-

import os
import sys
import argparse
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import gc
import cv2

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from dataset.data_loader2 import RSDataset, yolo_dataset_collate
# from dataset.data_loader import RSDataset
from model.DetGeo import DetGeo
from model.loss import yolo_loss, build_target, adjust_learning_rate, RRdata_loss
from utils.utils import AverageMeter, eval_iou_acc, decode_box, non_max_suppression
from utils.checkpoint import save_checkpoint, load_pretrain

from PIL import ImageDraw, ImageFont
from PIL import Image

def main():
    parser = argparse.ArgumentParser(
        description='cross-view object geo-localization')
    parser.add_argument('--gpu', default='3', help='gpu id')
    parser.add_argument('--num_workers', default=24, type=int, help='num workers for data loading')

    parser.add_argument('--max_epoch', default=25, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--emb_size', default=512, type=int, help='embedding dimensions')
    parser.add_argument('--img_size', default=1024, type=int, help='image size')
    parser.add_argument('--fea_size', default=64, type=int, help='image size')
    parser.add_argument('--data_root', type=str, default='./data', help='path to the root folder of all dataset')
    parser.add_argument('--data_name', default='RRdata', type=str, help='CVOGL_DroneAerial/CVOGL_SVI/RRdata')
    parser.add_argument('--pretrain', default=False, type=str, metavar='PATH')
    parser.add_argument('--print_freq', '-p', default=50, type=int, metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--savename', default='default', type=str, help='Name head for saved model')
    parser.add_argument('--seed', default=13, type=int, help='random seed')
    parser.add_argument('--beta', default=1.0, type=float, help='the weight of cls loss')
    parser.add_argument('--test', dest='test', default=True, action='store_true', help='test')
    parser.add_argument('--val', dest='val', default=False, action='store_true', help='val')
    parser.add_argument('--min_overlap', default=0.03, type=float, help='min overlap')
    parser.add_argument('--save_path', type=str, default='./detect_result/', help='path to the root folder of all dataset')

    global args, anchors_full
    args = parser.parse_args()
    print('----------------------------------------------------------------------')
    print(sys.argv[0])
    print(args)
    print('----------------------------------------------------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ## fix seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 2)
    torch.cuda.manual_seed_all(args.seed + 3)

    eps = 1e-10
    ## following anchor sizes calculated by kmeans under args.anchor_imsize=1024
    if args.data_name == 'CVOGL_DroneAerial':
        anchors = '37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573'
    elif args.data_name == 'CVOGL_SVI':
        anchors = '37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573'
    elif args.data_name == 'RRdata':
        anchors = '37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573'
    else:
        assert (False)
    args.anchors = anchors

    ## save logs
    if args.savename == 'default':
        # args.savename = '%s_batch%d' % (args.dataset, args.batch_size)
        args.savename = '%s_batch%d' % (args.data_root, args.batch_size)
    if not os.path.exists('./logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.INFO, filename="./logs/%s" % args.savename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(str(sys.argv))
    logging.info(str(args))

    input_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RSDataset(data_root=args.data_root,
                              data_name=args.data_name,
                              split_name='train',
                              img_size=args.img_size,
                              transform=input_transform,
                              augment=True)
    val_dataset = RSDataset(data_root=args.data_root,
                            data_name=args.data_name,
                            split_name='val',
                            img_size=args.img_size,
                            transform=input_transform)
    test_dataset = RSDataset(data_root=args.data_root,
                             data_name=args.data_name,
                             split_name='test',
                             img_size=args.img_size,
                             transform=input_transform)

    if args.data_name == 'RRdata':
        # collate_fn=yolo_dataset_collate,
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, drop_last=False, collate_fn=yolo_dataset_collate,
                                  num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                pin_memory=True, drop_last=False, collate_fn=yolo_dataset_collate,
                                num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 pin_memory=True, drop_last=False,collate_fn=yolo_dataset_collate, num_workers=args.num_workers)

    else:
        # collate_fn=yolo_dataset_collate,
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, drop_last=False,
                                  num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                pin_memory=True, drop_last=False,
                                num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 pin_memory=True, drop_last=False,collate_fn=yolo_dataset_collate, num_workers=args.num_workers)

    ## Model
    model = DetGeo(data_name=args.data_name)

    model = torch.nn.DataParallel(model).cuda()

    # if args.pretrain:
    #     model = load_pretrain(model, args, logging)

    model_path = "./weight/ep025-loss312.189.pth"
    """加载权重"""
    weight_dict = torch.load(model_path)  # 读取的网络权重字典
    model.load_state_dict(weight_dict)
    # 打印模型使用的GPU设备数量
    print(f"Number of GPUs used: {torch.cuda.device_count()}")


    print('Num of parameters:', sum([param.nelement() for param in model.parameters()]))
    logging.info('Num of parameters:%d' % int(sum([param.nelement() for param in model.parameters()])))

    optimizer = torch.optim.RMSprop([{'params': model.parameters()}, ], lr=args.lr, weight_decay=0.0005)

    ## training and testing
    best_accu = -float('Inf')

    if args.test:
        for epoch in range(args.max_epoch):
            adjust_learning_rate(args, optimizer, epoch)
            gc.collect()
            test_epoch(test_loader, model, args)





def test_epoch(data_loader, model, args):

    imgs_obj_N=[]
    imgs_obj_pairs=[]
    single_img_precions=[]

    torch.cuda.empty_cache()
    model.eval()
    anchors_full = np.array([float(x.strip()) for x in args.anchors.split(',')])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).cuda()

    for batch_idx, (query_imgs, rs_imgs, mat_clickxy, ori_gt_bbox, idxs) in enumerate(data_loader):

        ref_gt_bbox=[sublist[:4]  for sublist in ori_gt_bbox[0].tolist()] #选择列表的前4个元素
        query_gt_bbox = [sublist[4:8] for sublist in ori_gt_bbox[0].tolist()]

        query_imgs, rs_imgs = query_imgs.cuda(), rs_imgs.cuda()
        mat_clickxy = mat_clickxy.cuda()

        # ori_gt_bbox = ori_gt_bbox.cuda()
        # ori_gt_bbox = torch.clamp(ori_gt_bbox, min=0, max=args.img_size-1)

        with torch.no_grad():
            pred_anchor, attn_score = model(query_imgs, rs_imgs, mat_clickxy)
        pred_anchor = pred_anchor.view(pred_anchor.shape[0], 9, 10, pred_anchor.shape[2], pred_anchor.shape[3])

        """***********************不同数据集*******************************"""
        if args.data_name == 'RRdata':
            pred_anchor = pred_anchor.view(pred_anchor.shape[0], 9 , 10, pred_anchor.shape[3], pred_anchor.shape[3])
            #outputs=【batch，len（anchor）*w*h，7】
            ref_outputs = decode_box(pred_anchor[:,:,0:5,:,:], args.img_size, anchors_full)
            query_outputs = decode_box(pred_anchor[:,:,5:10,:,:], args.img_size, anchors_full)

            outputs=torch.cat((ref_outputs, query_outputs), dim=-1)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = non_max_suppression(outputs, conf_thres=0.0001,nms_thres=0.4)
            # query_outputs = non_max_suppression(ref_outputs, conf_thres=0.005, nms_thres=0.4)

            if results[0] is None:
                continue
            if len(results[0][:, 4])==1:
                top_conf = results[:, 4]
                top_boxes = results[:, :4]
                query_pre_boxes = results[:, 5:9]
            else:
                top_conf = results[0][:, 4]
                top_boxes = results[0][:, :4]
                query_pre_boxes = results[0][:, 5:9]


            # ---------------------------------------------------------#
            #   设置字体与边框厚度
            # ---------------------------------------------------------#
            #rs_imgs.shape=【batch，c，w，h】

            # 使用squeeze()方法压缩维度
            rs_imgs = rs_imgs.squeeze(0)
            # 将Tensor对象转换为NumPy数组
            rs_imgs = rs_imgs.detach().cpu().numpy()
            # 转换为Image对象
            rs_imgs = Image.fromarray(rs_imgs.transpose((1, 2, 0)).astype('uint8'))


            font = ImageFont.truetype(font='model/simhei.ttf',
                                      size=np.floor(3e-2 * rs_imgs.size[0] + 0.5).astype('int32'))
            thickness = int(max((rs_imgs.size[0] + rs_imgs.size[1]) // np.mean([1024,1024]), 1))

            # 读取指定路径的图像
            # 使用Image.open()函数打开图像文件
            rs_imgs = Image.open("./data/dataset/sat/" + str(idxs[0]) + ".jpg")

            #获取查询图像的所有框信息
            img_obj_pairs=0
            img_obj_N=len(top_boxes) #单张图像上目标的总数
            flag=0
            for i, ref_pre_box in list(enumerate(top_boxes)):
                ref_pre_box = ref_pre_box[:4]
                query_pre_box =query_pre_boxes[i]
                score = top_conf[i]
                top, left, bottom, right = ref_pre_box

                #与参考图像中当前目标预测框重合度最大的gt框
                reference_gt_box=max_overlap(ref_pre_box,ref_gt_bbox,args.min_overlap,mode="ref")
                #与查询图像中当前目标预测框重合度最大的gt框
                query_gt_box = max_overlap(query_pre_box, query_gt_bbox, args.min_overlap,mode="query")



                #获取当前图像对应的标签信息
                # if reference_gt_box==None or query_gt_box==None:
                #     continue
                # else:
                if reference_gt_box != None :
                    print("11111111111111111111111111111111")
                    # pre_result = reference_gt_box + query_gt_box
                    # if pre_result in ori_gt_bbox:
                    #     img_obj_pairs = img_obj_pairs + 1

                        # 检测结果可视化
                    top = max(0, np.floor(top).astype('int32'))
                    left = max(0, np.floor(left).astype('int32'))
                    bottom = min(rs_imgs.size[0], np.floor(bottom).astype('int32'))
                    right = min(rs_imgs.size[1], np.floor(right).astype('int32'))
                    predicted_class = "building"
                    label = '{} {:.2f}'.format(predicted_class, score)
                    draw = ImageDraw.Draw(rs_imgs)
                    label_size = draw.textsize(label, font)
                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])
                    label = label.encode('utf-8')
                    print(left, top, right, bottom)
                    draw.rectangle([left, top, right, bottom], outline=(255, 0, 0))

                    draw.text(text_origin, str("目标编号N"), fill=(0, 0, 0), font=font)
                    rs_imgs = rs_imgs
                    flag=1





            single_img_precion=img_obj_pairs/img_obj_N
            single_img_precions.append(single_img_precion)
            imgs_obj_N.append(img_obj_N)
            imgs_obj_pairs.append(img_obj_pairs)
            if flag==1:
                # 调用save()方法保存图像
                rs_imgs.save(args.save_path + str(idxs[0]) + ".jpg")


    imgs_precion = sum(imgs_obj_pairs) / sum(imgs_obj_N) * 1.0
    mean_single_img_precion=sum(single_img_precions)/len(single_img_precion)*1.0
    print(imgs_precion,mean_single_img_precion)






def max_overlap(input_box,gt_boxes,min_overlap,mode):
    if mode =="ref":
        #此时的gt_boxes存储形式为：x1，y1，x2，y2，是相对1024×1024的坐标值,input_box的存储形式也是x1，y1，x2，y2，是相对1024×1024的坐标值
        #需要将其转化到750×750的图像上
        #此时的input_box是相对于图像尺寸
        ovmax = -1
        get_match_box = None
        for gt_box in gt_boxes:
            bi = [max(input_box[0], gt_box[0]), max(input_box[1], gt_box[1]), min(input_box[2], gt_box[2]),
                  min(input_box[3], gt_box[3])]
            iw = bi[2] - bi[0] + 1
            ih = bi[3] - bi[1] + 1
            if iw > 0 and ih > 0:
                ua = (input_box[2] - input_box[0] + 1) * (input_box[3] - input_box[1] + 1) + (
                            gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1) - iw * ih
                ov = iw * ih / ua
                if ov > ovmax and ov > min_overlap:
                    ovmax = ov
                    get_match_box = gt_box
        return get_match_box














if __name__ == "__main__":
    main()


