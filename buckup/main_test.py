
import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append("mmocr_text")
sys.path.append("lama")

from utils import edge_blur, aline_text_color, post_proc_image
import numpy as np
import cv2
from lama.lama_interface import LamaInterface
from PIL import Image
from segment_anything import SAMInterface
from mmocr_text.mmocr.apis import TextDetInferencer



# 配置
ref_img = "1690784723456_3"
save_path = "output"
dbnetpp_path = "mmocr_text/ckpts/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth"
sam_path = "segment_anything/checkpoints/sam_vit_h_4b8939.pth"
lama_path = "lama/big_lama/models/best.ckpt"
lama_config = "lama/big_lama/config.yaml"


# 通过DBNet++检测文本区域
inferencer = TextDetInferencer(model='DBNetPP', weights=dbnetpp_path)
poly_list = inferencer(f'./example/{ref_img}_ref.png')['predictions'][0]['polygons']


# 通过 SAM模型 获取更为精准的文本分割结果
sam_interface = SAMInterface(model_type='vit_h', ckpt_path=sam_path)
mask = sam_interface.infer(poly_list, f'./example/{ref_img}_ref.png')
cv2.imwrite(f'{save_path}/mask_{ref_img}.png', mask*255)

# 使用 Lama模型 填充A中崩坏的文本区域
lama = LamaInterface(lama_path, lama_config)
img = Image.open(f'./example/{ref_img}.png')
mask = Image.open(f'{save_path}/mask_{ref_img}.png')  # 黑白mask，填充白色区域
pred = lama.infer(img, mask)
pred.save(f'{save_path}/lama_{ref_img}.png')

# 将完好的文本粘贴到对应区域


def infer(image_lama: Image.Image, image_ref: Image.Image, mask: np.array) -> Image.Image:
    """
    根据文本清晰的参考图修复文本
    :param image_lama : lama抹除文本畸变的图像
    :param image_ref: 具有清晰文本的参考图像(mixpipe出来的图)
    :param mask: 需要替换区域的mask
    :return: 将image_ref的mask区域替换到image_lama上
    """
    image = image_lama.convert('RGB')
    image_ref = image_ref.convert('RGB')

    w, h = image.size
    # 强制参考图像和被替换图像相同尺寸
    image_ref = image_ref.resize((w, h), Image.ANTIALIAS)

    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    image_ref = cv2.cvtColor(np.asarray(image_ref), cv2.COLOR_RGB2BGR)
    mask = np.array(mask)
    image_ref = aline_text_color(image, image_ref, mask > 127)

    # 边缘虚化
    image_ref_a = np.concatenate((image_ref, mask[:, :, None]), axis=2)
    image_ref_a = edge_blur(image_ref_a)
    mask = (image_ref_a[:, :, 3]/255)[:, :, None]

    image = (image*(1-mask) + image_ref*mask).astype(np.uint8)
    pred = post_proc_image(image)
    return pred


img_lama = Image.open(F'{save_path}/lama_{ref_img}.png')
img_ref = Image.open(f'./example/{ref_img}_ref.png')
mask = Image.open(f'{save_path}/mask_{ref_img}.png') 

pred = infer(img_lama, img_ref, mask)
pred.save(f'{save_path}/refine_{ref_img}.png')
