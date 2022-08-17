# -*- coding: utf-8 -*-
#gitclon
"""

# Commented out IPython magic to ensure Python compatibility.
## MMSegmentation Framework을 사용하기 위한 Install
# Install PyTorch
!pip install torch==1.12.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
# Install MMCV
!pip install openmim
!mim install mmcv-full==1.6.0
# git clone
!git clone https://github.com/open-mmlab/mmsegmentation.git 
# %cd mmsegmentation
!pip install -e .

"""# Deeplav3_unet"""

!gdown 1OC7HRDEp99HiMdVD8p1P1HWDcl3My0eX
!unzip /content/mmsegmentation/DeepLabUnet.zip

"""# 512"""

!gdown 1vAfGl3bDwx7HS2N8eqYnzG8e452IV3nj
!unzip /content/mmsegmentation/Multi_512x512_MMSeg.zip

"""# 256"""

!gdown 1787fZejOECLSAWpMg8fDwDgWSz6gTusT # multi class 256x256
!unzip /content/mmsegmentation/Multi_256x256_MMSeg.zip

"""# original"""

!gdown 1VnXyPKFiRL5Cvmw3UXLIK6Id93vjgKW5
!unzip /content/mmsegmentation/hubmap-organ-segmentation.zip

"""#wamdb"""

!pip install -q --upgrade wandb

import wandb
wandb.login()

!mkdir -p ./mmseg_data/splits

# !mkdir /content/mmsegmentation/mmseg_data
!mv /content/mmsegmentation/train /content/mmsegmentation/mmseg_data
!mv /content/mmsegmentation/masks /content/mmsegmentation/mmseg_data

"""##### Start with 256x256 data"""

from glob import glob
import numpy as np
import cv2
import os
from sklearn.model_selection import StratifiedKFold

Fold = 10
all_mask_files = glob("./mmseg_data/masks/*")
masks = []
num_mask = np.zeros((6,Fold))

for i in range(len(all_mask_files)):
    mask = cv2.imread(all_mask_files[i])
    masks.append(mask.max())

## train_valid_split, KFold : 교차검증
split = list(StratifiedKFold(n_splits=Fold, shuffle=True, random_state=2022).split(all_mask_files, masks))
for fold, (train_idx, valid_idx) in enumerate(split):
    for i in valid_idx:
        num_mask[masks[i]]+=1
    with open(f"./mmseg_data/splits/fold_{fold}.txt", "w") as f:
        for idx in train_idx:
            f.write(os.path.basename(all_mask_files[idx])[:-4] + "\n")
    with open(f"./mmseg_data/splits/valid_{fold}.txt", "w") as f:
        for idx in valid_idx:
            f.write(os.path.basename(all_mask_files[idx])[:-4] + "\n")
print(num_mask)

"""* Larger backbone and more iters may lead to better score!"""

## config.py 생성
!touch /content/mmsegmentation/config.py

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# 
# cat <<EOT >> /content/mmsegmentation/config.py # 빈 config.py 생성해주기
# 
# ## model setting
# norm_cfg = dict(type='SyncBN', requires_grad=True)
# model = dict(
#     type='EncoderDecoder',
#     pretrained=None,
#     backbone=dict(
#         type='UNet',
#         in_channels=3,
#         base_channels=64,
#         num_stages=5,
#         strides=(1, 1, 1, 1, 1),
#         enc_num_convs=(2, 2, 2, 2, 2),
#         dec_num_convs=(2, 2, 2, 2),
#         downsamples=(True, True, True, True),
#         enc_dilations=(1, 1, 1, 1, 1),
#         dec_dilations=(1, 1, 1, 1),
#         with_cp=False,
#         conv_cfg=None,
#         norm_cfg=norm_cfg,
#         act_cfg=dict(type='ReLU'),
#         upsample_cfg=dict(type='InterpConv'),
#         norm_eval=False),
#     decode_head=dict(
#         type='FCNHead',
#         in_channels=64,
#         in_index=4,
#         channels=64,
#         num_convs=1,
#         concat_input=False,
#         dropout_ratio=0.1,
#         num_classes=6,
#         norm_cfg=norm_cfg,
#         align_corners=False,
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
#     auxiliary_head=dict(
#         type='FCNHead',
#         in_channels=128,
#         in_index=3,
#         channels=64,
#         num_convs=1,
#         concat_input=False,
#         dropout_ratio=0.1,
#         num_classes=6,
#         norm_cfg=norm_cfg,
#         align_corners=False,
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
#     train_cfg=dict(),
#     test_cfg=dict(mode='whole'))
# 
# ## dataset settings
# dataset_type = 'CustomDataset'
# data_root = '/content/mmsegmentation/mmseg_data/'
# classes = ['background', 'kidney', 'prostate', 'largeintestine', 'spleen', 'lung']
# palette = [[0,0,0], [255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255]]
# img_norm_cfg = dict(mean=[196.869, 190.186, 194.802], std=[63.010, 66.765, 65.745], to_rgb=True)
# size = 256
# 
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),
#     dict(type='Resize', img_scale=(size, size), keep_ratio=True),
#     dict(type='RandomFlip', prob=0.5, direction='horizontal'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=(size, size), pad_val=0, seg_pad_val=255),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(size, size),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size=(size, size), pad_val=0, seg_pad_val=255),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# data = dict(
#     samples_per_gpu=8,
#     workers_per_gpu=4,
#     train=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='train', # image경로
#         ann_dir='masks', # mask.png경로
#         img_suffix=".png",
#         seg_map_suffix='.png',
#         split="splits/fold_0.txt",
#         classes=classes,
#         palette=palette,
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='train',
#         ann_dir='masks',
#         img_suffix=".png",
#         seg_map_suffix='.png',
#         split="splits/valid_0.txt",
#         classes=classes,
#         palette=palette,
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         data_root=data_root,
#         test_mode=True,
#         img_dir='train',
#         ann_dir='masks',
#         img_suffix=".png",
#         seg_map_suffix='.png',
#         classes=classes,
#         palette=palette,
#         pipeline=test_pipeline))
# 
# # yapf:disable
# log_config = dict(
#     interval=50,
#     hooks=[
#         dict(type='TextLoggerHook', by_epoch=False),
#         dict(type='WandbLoggerHook', interval=50, # connect Wandb
#              init_kwargs=dict(
#                  project='HuBMAP',
#                  entity='yoonsomi',
#                  name='model_name'),
#              )
#     ])
# # yapf:enable
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# load_from = None
# resume_from = None
# workflow = [('train', 1)]
# cudnn_benchmark = True
# 
# total_iters = 5000
# # optimizer
# optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05)
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# # learning policy
# lr_config = dict(policy='poly',
#                  warmup='linear',
#                  warmup_iters=500,
#                  warmup_ratio=1e-6,
#                  power=1.0, min_lr=0.0, by_epoch=False)
# # runtime settings
# find_unused_parameters = True
# runner = dict(type = 'IterBasedRunner', max_iters = total_iters)
# checkpoint_config = dict(by_epoch=False, interval=-1, save_optimizer=False)
# evaluation = dict(by_epoch=False, interval=500, metric='mDice', pre_eval=True)
# fp16 = dict()
# work_dir = './baseline'
# EOT

"""#### Train with api"""

!pwd

!python /content/mmsegmentation/tools/train.py /content/mmsegmentation/Deep_Unet/deeplabv3_unet_s5-d16_256x256_40k_hrf.py

"""#inference"""

import cv2
import numpy as np
import pandas as pd
import os
from glob import glob
from tqdm.notebook import tqdm
import sys
import gc
sys.path.append('./mmsegmentation-master')
from mmseg.apis import init_segmentor, inference_segmentor
from mmcv.utils import config

"""경로 바꿔주기"""

configs = [
    '/content/work_dirs/deeplabv3_unet_s5-d16_256x256_40k_hrf/deeplabv3_unet_s5-d16_256x256_40k_hrf.py',
]
ckpts = [
    '/content/work_dirs/deeplabv3_unet_s5-d16_256x256_40k_hrf/epoch_10.pth' # ckpts.pth(weight) 경로
]

DATA = '/content/mmsegmentation/organ/' # test img 경로
df_sample = pd.read_csv('/content/mmsegmentation/sample_submission.csv').set_index('id') # sample_submission.csv(?) 경로

models = []
for idx,(cfg, ckpt) in enumerate(zip(configs, ckpts)):
    cfg = config.Config.fromfile(cfg)
    model = init_segmentor(cfg, ckpt, device='cuda:0')
    models.append(model)

# configs = [
#     '/content/mmsegmentation/config.py',
# ]
# ckpts = [
#     '/content/baseline/latest.pth' # ckpts.pth(weight) 경로
# ]

# DATA = '/content/mmsegmentation/organ/' # test img 경로
# df_sample = pd.read_csv('/content/mmsegmentation/sample_submission.csv').set_index('id') # sample_submission.csv(?) 경로

# models = []
# for idx,(cfg, ckpt) in enumerate(zip(configs, ckpts)):
#     cfg = config.Config.fromfile(cfg)
#     model = init_segmentor(cfg, ckpt, device='cuda:0')
#     models.append(model)

### mask -> rle
def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

names,preds = [],[]
imgs, pd_mks = [],[]
debug = len(df_sample)<2
for idx,row in tqdm(df_sample.iterrows(),total=len(df_sample)):
    img  = cv2.imread(os.path.join(DATA,str(idx)+'.tiff'))
    im_  = img/img.max()
    pred = inference_segmentor(models[0], img)[0]
    pred = (pred>0).astype(np.uint8)
    rle = rle_encode_less_memory(pred)
    names.append(str(idx))
    preds.append(rle)
    if debug:
        imgs.append(img)
        pd_mks.append(pred)
    del img, pred, rle, idx, row
    gc.collect()

if debug:
    import matplotlib.pyplot as plt
    for img, mask in zip(imgs, pd_mks):
        plt.figure(figsize=(12, 7))
        plt.subplot(1, 3, 1); plt.imshow(img); plt.axis('OFF'); plt.title('image')
        plt.subplot(1, 3, 2); plt.imshow(mask*255); plt.axis('OFF'); plt.title('mask')
        plt.subplot(1, 3, 3); plt.imshow(img); plt.imshow(mask*255, alpha=0.4); plt.axis('OFF'); plt.title('overlay')
        plt.tight_layout()
        plt.show()

df = pd.DataFrame({'id':names,'rle':preds})
df.to_csv('submission.csv',index=False)

