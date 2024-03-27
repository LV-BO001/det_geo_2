
import os
import torch

# data_name = "CVOGL_SVI"
#
# split_name = "train"
# data_dir = "../data/"+data_name
#
#
# data_path = os.path.join(data_dir, '{0}_{1}.pth'.format(data_name, split_name))
# data_list = torch.load(data_path)
#
# for i in range(0,len(data_list)):
#     print(data_list[i])
#     _, queryimg_name, rsimg_name, _, click_xy, bbox, _, cls_name = data_list[i]

import numpy as np
            #xp1, xt1, yp1, yt1, wp1, wt1, hp1, ht1.
# A = np.array([[1,  -1,  0,   0,   0,   0,   0,    0],
#               [0,   0,  1,  -1,   0,   0,   0,    0],
#               [0,   0,  0,   0,  1,  -1,   0,    0],
#               [0,   0,  0,  0,   0,    0,   1,  -1,],
#               ])
#
# # 将增广矩阵进行高斯消元
# rref, pivots = np.linalg.qr(A)
#
# # 检查矩阵的秩
# rank = np.linalg.matrix_rank(rref)
#
# if rank == A.shape[1]:
#     print("齐次线性方程组的唯一解是零解")
# else:
#     print("齐次线性方程组有非零解")
input_shape=[1024,1024]
image_shape=[750,750]

x=np.concatenate([input_shape, image_shape], axis=-1)
print(x)