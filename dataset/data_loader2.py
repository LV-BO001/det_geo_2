# -*- coding: utf-8 -*-





"""用于加载多对多数据集"""



import os
import sys
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations
from shapely.geometry import Polygon
import xml.etree.ElementTree as ET
cv2.setNumThreads(0)
from PIL import Image
import torchvision.transforms as transforms


class DatasetNotFoundError(Exception):
    pass


class MyAugment:
    def __init__(self) -> None:
        self.transform = albumentations.Compose([
            albumentations.Blur(p=0.01),
            albumentations.MedianBlur(p=0.01),
            albumentations.ToGray(p=0.01),
            albumentations.CLAHE(p=0.01),
            albumentations.RandomBrightnessContrast(p=0.0),
            albumentations.RandomGamma(p=0.0),
            albumentations.ImageCompression(quality_lower=75, p=0.0)])

    def augment_hsv(self, im, hgain=0.5, sgain=0.5, vgain=0.5):
        # HSV color-space augmentation
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
            dtype = im.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=im)  # no return needed

    def __call__(self, img, bbox):
        imgh, imgw, _ = img.shape
        x, y, w, h = (bbox[0] + bbox[2]) / 2 / imgw, (bbox[1] + bbox[3]) / 2 / imgh, (bbox[2] - bbox[0]) / imgw, (
                    bbox[3] - bbox[1]) / imgh
        img = self.transform(image=img)['image']
        # self.augment_hsv(img)
        # Flip up-down
        if random.random() < 0.5:
            img = np.flipud(img)
            y = 1 - y

        # Flip left-right
        if random.random() < 0.5:
            img = np.fliplr(img)
            x = 1 - x
        #
        new_imgh, new_imgw, _ = img.shape
        assert new_imgh == imgh, new_imgw == imgw
        x, y, w, h = x * imgw, y * imgh, w * imgw, h * imgh

        # Crop image
        iscropped = False
        if random.random() < 0.5:
            left, top, right, bottom = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            if left >= new_imgw / 2:
                start_cropped_x = random.randint(0, int(0.15 * new_imgw))
                img = img[:, start_cropped_x:, :]
                left, right = left - start_cropped_x, right - start_cropped_x
            if right <= new_imgw / 2:
                start_cropped_x = random.randint(int(0.85 * new_imgw), new_imgw)
                img = img[:, 0:start_cropped_x, :]
            if top >= new_imgh / 2:
                start_cropped_y = random.randint(0, int(0.15 * new_imgh))
                img = img[start_cropped_y:, :, :]
                top, bottom = top - start_cropped_y, bottom - start_cropped_y
            if bottom <= new_imgh / 2:
                start_cropped_y = random.randint(int(0.85 * new_imgh), new_imgh)
                img = img[0:start_cropped_y, :, :]
            cropped_imgh, cropped_imgw, _ = img.shape
            left, top, right, bottom = left / cropped_imgw, top / cropped_imgh, right / cropped_imgw, bottom / cropped_imgh
            if cropped_imgh != new_imgh or cropped_imgw != new_imgw:
                img = cv2.resize(img, (new_imgh, new_imgw))
            new_cropped_imgh, new_cropped_imgw, _ = img.shape
            left, top, right, bottom = left * new_cropped_imgw, top * new_cropped_imgh, right * new_cropped_imgw, bottom * new_cropped_imgh
            x, y, w, h = (left + right) / 2, (top + bottom) / 2, right - left, bottom - top
            iscropped = True
        # if iscropped:
        #    print((new_imgw, new_imgh))
        #    print((cropped_imgw, cropped_imgh), flush=True)
        #    print('============')
        # print(type(img))
        # draw_bbox = np.array([x-w/2, y-h/2, x+w/2, y+h/2], dtype=int)
        # print(('draw_bbox', iscropped, draw_bbox), flush=True)
        # img_new=draw_rectangle(img, draw_bbox)
        # cv2.imwrite('tmp/'+str(random.randint(0,5000))+"_"+str(iscropped)+".jpg", img_new)

        new_bbox = [(x - w / 2), y - h / 2, x + w / 2, y + h / 2]
        # print(bbox)
        # print(new_bbox)
        # print('---end---')
        return img, np.array(new_bbox, dtype=int)


BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, x_max, y_min, y_max = int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])
    print(bbox, flush=True)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


class RSDataset(Dataset):
    def __init__(self, data_root, data_name='CVOGL', split_name='test', img_size=1024,
                 transform=None, augment=False):
        self.data_root = data_root
        self.data_name = data_name
        self.img_size = img_size
        self.transform = transform
        self.split_name = split_name
        self.augment = augment

        self.myaugment = MyAugment()

        if self.data_name == 'CVOGL_DroneAerial':
            data_dir = os.path.join(data_root, self.data_name)
            self.data_path = os.path.join(data_dir, '{0}_{1}.pth'.format(self.data_name, split_name))
            self.data_list = torch.load(self.data_path)
            self.queryimg_dir = os.path.join(data_dir, 'query')
            self.rsimg_dir = os.path.join(data_dir, 'satellite')
            self.rs_wh = 1024
            self.query_featuremap_hw = (256, 256)  # 52 #32
        elif self.data_name == 'CVOGL_SVI':
            data_dir = os.path.join(data_root, self.data_name)
            data_path = os.path.join(data_dir, '{0}_{1}.pth'.format(self.data_name, split_name))
            self.data_list = torch.load(data_path)
            self.queryimg_dir = os.path.join(data_dir, 'query')
            self.rsimg_dir = os.path.join(data_dir, 'satellite')
            self.rs_wh = 1024
            self.query_featuremap_hw = (256, 512)

        elif self.data_name =="RRdata":
            # data_dir = os.path.join(data_root, self.data_name)
            query_data_path = os.path.join(data_root, '{0}_{1}.txt'.format("panos",split_name ))
            reference_data_path = os.path.join(data_root, '{0}_{1}.txt'.format("sat", split_name))
            self.rs_wh = 1024
            self.query_featuremap_hw = (256, 512)


            # 加载卫星图像地址
            with open(query_data_path) as f1:
                self.query_train_lines = f1.readlines()

            with open(reference_data_path) as f2:
                self.refere_val_lines = f2.readlines()

            # self.data_list = torch.load(data_path)
            # self.queryimg_dir = os.path.join(data_dir, 'query')
            # self.rsimg_dir = os.path.join(data_dir, 'satellite')

        else:
            assert (False)

        self.rs_transform = albumentations.Compose([
            albumentations.RandomSizedBBoxSafeCrop(width=self.rs_wh, height=self.rs_wh, erosion_rate=0.2, p=0.2),
            albumentations.RandomRotate90(p=0.5),
            albumentations.GaussNoise(p=0.5),
            albumentations.HueSaturationValue(p=0.3),
            albumentations.OneOf([
                albumentations.Blur(p=0.4),
                albumentations.MedianBlur(p=0.3),
            ], p=0.5),
            albumentations.OneOf([
                albumentations.RandomBrightnessContrast(p=0.4),
                albumentations.CLAHE(p=0.3),
            ], p=0.5),
            albumentations.ToGray(p=0.2),
            albumentations.RandomGamma(p=0.3), ], bbox_params=albumentations.BboxParams(format='pascal_voc'))

    def __len__(self):
        return len(self.query_train_lines)

        # 将xml文件中的bb框转化为坐标点写进文件
    def get_bbox(self,image_name,panos_or_sat):
        data_path = 'data/dataset'
        print(os.path.join(data_path, '%s_xml/%s.xml' % (panos_or_sat, image_name)))
        in_file = open(os.path.join(data_path, '%s_xml/%s.xml' % (panos_or_sat, image_name)),
                           encoding='utf-8')
        box = []
        tree = ET.parse(in_file)
        root = tree.getroot()
        objects = root.findall("object")
        # 枚举xml中objects的数目，并选择相应的bb框
        for i, obj in enumerate(objects):
            bbox = obj.find('bndbox')

            b = [int(float(bbox.find('xmin').text)), int(float(bbox.find('ymin').text)),
                     int(float(bbox.find('xmax').text)), int(float(bbox.find('ymax').text))]
            box.append(b)
        return box

    def get_data(self, annotation_line,panos_or_sat):
        # self.list.append(box_index)
        # self.list.append(annotation_line)

        # print("self.list*****************__",self.list)
        # print("line______",annotation_line)
        line = annotation_line.split()
        #获取图像的名字
        #line内容为：/raid/lvbo/object2object_retrieval/dataset/panos/0000002.jpg 1078,109,1201,140
        image_name=line[0].split("/")[-1].split(".")[0]
        #获取图像内的所有目标框
        # box=self.get_bbox(image_name,panos_or_sat)

        # box=line[1:]
        new_box = []
        for i in range(1,len(line)):
            new_list = [int(x)  for x in line[i].split(",")]
            new_box.append(new_list)




        # ------------------------------#
        #   读取图像并转换成RGB图像
        # ------------------------------#
        image_data = Image.open(line[0])
        # 定义转换函数
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # 将图像转换为张量
        # image_data = transform(image_data)
        # ------------------------------#
        #   获得预测框
        # ------------------------------#

        # print("new_box_____1", line[1:],type(line),len(line))

        # new_box = [list(map(int, box.split(','))) for box in line[1:]]  # 将 box 变量转换为列表对象
        # print("new_box",new_box,len(new_box),box_index_ture)

        return image_name,image_data, new_box



    def __getitem__(self, idx):

        # 全景图
        query_image_name,query_image, query_bbox = self.get_data(self.query_train_lines[idx], panos_or_sat="panos")
        ref_image_name,ref_image, ref_bbox = self.get_data(self.refere_val_lines[idx], panos_or_sat="sat")
        if len(ref_bbox) != len(query_bbox):
            print("错误！！！！！！！")
            print(ref_image_name,query_bbox,ref_bbox)
            return  None

        # query_bbox= self.get_bbox(image_name,self.data_path,panos_or_sat="panos")
        # ref_bbox = self.get_bbox(image_name,self.data_path,panos_or_sat="sat")


        ## box format: to x1y1x2y2
        # query_bbox = np.array(query_bbox, dtype=int)

        # queryimg = cv2.imread(os.path.join(self.queryimg_dir, image_name))
        # queryimg = cv2.cvtColor(queryimg, cv2.COLOR_BGR2RGB)
        #
        # rsimg = cv2.imread(os.path.join(self.rsimg_dir, image_name))
        # rsimg = cv2.cvtColor(rsimg, cv2.COLOR_BGR2RGB)
        # if self.augment:
        #     rs_transformed = self.rs_transform(image=rsimg, bboxes=[list(bbox) + [cls_name]])
        #     rsimg = rs_transformed['image']
        #     bbox = rs_transformed['bboxes'][0][0:4]

        mat_query = np.zeros((query_image.size[0], query_image.size[1]), dtype=np.float32)
        if len(query_bbox)==1:
            # print("""1111""")
            # print("len(query_bbox)==1________________query_bbox[2]___________",query_image_name, query_bbox,query_bbox[0])
            # print("""22222""")
            for m in range(query_bbox[0][0], query_bbox[0][2]):

                for n in range(query_bbox[0][1], query_bbox[0][3]):
                    mat_query[m, n] = 1.0
        else:
            # print("len(query_bbox)==2________________query_bbox[2]___________", query_bbox)
            for i in range(0, len(query_bbox)):
                for m in range(query_bbox[i][0], query_bbox[i][2]):
                    for n in range(query_bbox[i][1], query_bbox[i][3]):
                        mat_query[m, n] = 1.0

        # 创建PIL图像对象
        mat_query = Image.fromarray(mat_query)
        # 调整图像大小
        mat_query = mat_query.resize((512, 256))

        query_image = query_image.resize((512, 256))

        ref_image = ref_image.resize((1024, 1024))

        # mat_referrence = np.zeros((ref_image.shape[1], ref_image.shape[2]), dtype=np.float32)

        if len(ref_bbox)==1:
            bbox = []
            ref_bbox = [x * 1024/750.0 for x in ref_bbox[0]]

            query_x1= query_bbox[0][0]*512/1232.0     #x对应1232，y对应224
            query_x2 = query_bbox[0][2] * 512 / 1232.0
            query_y1 = query_bbox[0][1] * 256 / 224.0
            query_y2 = query_bbox[0][3] * 256 / 224.0

            ref_bbox.append(query_x1)
            ref_bbox.append(query_y1)
            ref_bbox.append(query_x2)
            ref_bbox.append(query_y2)

            # x_center=(query_bbox[0][0]+query_bbox[0][2])/2.0
            # y_center = (query_bbox[0][1] + query_bbox[0][3])/ 2.0
            # ref_bbox.append(x_center)
            # ref_bbox.append(y_center)

            bbox.append(ref_bbox)
        else:
            bbox = []
            # if len(ref_bbox)!=len(query_bbox):
            #     print("错误！！！！！！！")
            for i in range(0, len(ref_bbox)):

                scaled_sublist = [x * 1024/750.0 for x in ref_bbox[i]]

                query_x1 = query_bbox[i][0] * 512 / 1232.0  # x对应1232，y对应224
                query_x2 = query_bbox[i][2] * 512 / 1232.0
                query_y1 = query_bbox[i][1] * 256 / 224.0
                query_y2 = query_bbox[i][3] * 256 / 224.0

                scaled_sublist.append(query_x1)
                scaled_sublist.append(query_y1)
                scaled_sublist.append(query_x2)
                scaled_sublist.append(query_y2)

                # print(query_bbox)
                # x_center = (query_bbox[i][0] + query_bbox[i][2])/ 2.0
                # y_center = (query_bbox[i][1] + query_bbox[i][3])/ 2.0
                # scaled_sublist.append(x_center)
                # scaled_sublist.append(y_center)
            bbox.append(scaled_sublist)



                # for sublist in ref_bbox:
                #     scaled_sublist = []
                #     for x in sublist:
                #         scaled_sublist.append(x * 1.365)
                #         x_center = query_bbox[0] + query_bbox[1] / 2.0
                #         y_center = query_bbox[2] + query_bbox[3] / 2.0
                #
                #         scaled_sublist.append(x_center)
                #         scaled_sublist.append(y_center)
                #
                #     bbox.append(scaled_sublist)
                # for m in range(ref_bbox[i][1], ref_bbox[i][3]):
                #     for n in range(ref_bbox[i][0], ref_bbox[i][2]):
                #         mat_referrence[m, n] = 1.0

        # 定义图像转换
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        mat_query = transform(mat_query)
        # Norm, to tensor
        if self.transform is not None:

            rsimg = self.transform(ref_image.copy())

            queryimg = self.transform(query_image.copy())



        # print("bbox长度",len(bbox))
        mat_query = torch.squeeze(mat_query)
        return queryimg, rsimg,mat_query, np.array(bbox, dtype=np.float32), query_image_name

# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):

    #batch:batch个查询图像、参考图像、查询矩阵、bbox、图像id
    queryimgs = []
    rsimgs = []
    mat_querys = []
    bboxs = []
    idxs = []


    for queryimg, rsimg,mat_query, bbox, idx in batch:
        queryimg=np.array(queryimg)
        rsimg = np.array(rsimg)
        mat_query = np.array(mat_query)

        queryimgs.append(queryimg)
        rsimgs.append(rsimg)
        mat_querys.append(mat_query)
        bboxs.append(bbox)
        idxs.append(idx)
    queryimgs = torch.from_numpy(np.array(queryimgs)).type(torch.FloatTensor)
    rsimgs = torch.from_numpy(np.array(rsimgs)).type(torch.FloatTensor)
    mat_querys = torch.from_numpy(np.array(mat_querys)).type(torch.FloatTensor)
    bboxs = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxs]
    # bboxs = np.array(bboxs, dtype=np.float32)
    # bboxs = torch.from_numpy(bboxs)

    return queryimgs, rsimgs,mat_querys, bboxs, idxs