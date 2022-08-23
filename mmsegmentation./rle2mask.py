import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tifffile as tiff 
from tqdm.auto import tqdm

plt.style.use("Solarize_Light2")

BASE_PATH = "/content/hub_organ/"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
### 경로, 부경로 지정해주기

df = pd.read_csv(
    os.path.join(BASE_PATH, "train.csv")
)
df.head()
### train csv file 불러오기

img_id_1 = 10044
### 이미지 번호임 첫번쨰
img_1 = tiff.imread(BASE_PATH + "train_images/" + str(img_id_1) + ".tiff")
### tiff파일 읽어오기, 지정해둔 경로에 파일 이름을 주고 확장자까지 입력
print(img_1.shape)

plt.figure(figsize=(15, 15))
### 이건 뽑아볼 사진 사이즈임
plt.imshow(img_1)
plt.axis("off")
### axis를 제거하는거임 아니면 표 모양 그대로 나옴(미관용이라고 함)

def mask2rle(img):
    pixels= img.T.flatten()
    ### image array 펴주기
    pixels = np.concatenate([[0], pixels, [0]])
    ### 양 옆에 0값을 넣어주기
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    ### np.where 조건 만족하는 위치 인덱스 찾기
    runs[1::2] -= runs[::2]
    ### 슬라이싱을 하기 위해서 양 옆에 0을 주었던거임
    return ' '.join(str(x) for x in runs)
 
def rle2mask(mask_rle, shape=(1600,256)):
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle2mask(mask_rle, shape=(1600,256)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

!mkdir /content/mask_img

from PIL import Image

for idx in range(len(df)):
  ### pd를 가져옴
    img_id = df.loc[idx]['id']
    ### 행으로 뽑기
    img_h = df.loc[idx]['img_height']
    img_w = df.loc[idx]['img_width']
    img_rle = df.loc[idx]['rle']
    ###iloc은 열뽑기


    mask_img = rle2mask(img_rle, shape=(img_w, img_h))
    mask_img = Image.fromarray(mask_img)
    mask_img.save(f'/content/mask_img/{img_id}.png')
    ### 경로에 image id포함 시켜서 폴더에 담기

# mask_1 = rle2mask(df[df["id"]==img_id_1]["rle"].iloc[-1], (img_1.shape[1], img_1.shape[0]))
# ### 
# mask_1.shape

mask_1 = rle2mask(df[df["id"]==img_id_1]["rle"].iloc[-1], (img_1.shape[1], img_1.shape[0]))
mask_1.shape

mask = Image.open('/content/mask_img/10044.png')
### 이미지 경로 불러오기
mask = np.array(mask)
### array형식으로 변경 해주기
plt.imshow(mask * 255)
### mask가 안보이니 곱해주기
plt.axis("off")

plt.figure(figsize=(10,10))
# plt.imshow(img_1) ### image랑 같이 뽑으려면 주석 풀기
plt.imshow(mask_1, cmap='hot', alpha=0.5) ##cmap='coolwarm' 색 지정 한거임
plt.axis("off")
### image, mask 겹쳐서뽑기

