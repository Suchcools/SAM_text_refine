import os
import sys
sys.path.append("mmocr_text")
sys.path.append("lama")

import warnings
warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from mmocr_text.mmocr.apis import TextDetInferencer
from segment_anything import SAMInterface
from lama.lama_interface import LamaInterface
from utils import edge_blur, aline_text_color, post_proc_image

import numpy as np
import cv2
from PIL import Image


def detect_text_area(img_path, model, weights):
    inferencer = TextDetInferencer(model=model, weights=weights)
    return inferencer(img_path)['predictions'][0]['polygons']


def get_text_mask(poly_list, img_path, model_type, ckpt_path):
    sam_interface = SAMInterface(model_type=model_type, ckpt_path=ckpt_path)
    return sam_interface.infer(poly_list, img_path)


def fill_text_using_lama(img_path, mask_path, lama_path, lama_config):
    lama = LamaInterface(lama_path, lama_config)
    img = Image.open(img_path)
    mask = Image.open(mask_path)
    return lama.infer(img, mask)


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


def main():
    # Configurations
    ref_img = "1690952943765_1"
    save_path = "output"
    dbnetpp_path = "mmocr_text/ckpts/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth"
    sam_path = "segment_anything/checkpoints/sam_vit_h_4b8939.pth"
    lama_path = "lama/big_lama/models/best.ckpt"
    lama_config = "lama/big_lama/config.yaml"

    # Detect text area using DBNet++
    poly_list = detect_text_area(f'./example/{ref_img}_ref.png', 'DBNetPP', dbnetpp_path)

    # Get precise text mask using SAM model
    mask = get_text_mask(poly_list, f'./example/{ref_img}_ref.png', 'vit_h', sam_path)
    cv2.imwrite(f'{save_path}/mask_{ref_img}.png', mask*255)

    # Fill corrupted text areas using Lama model
    pred_lama = fill_text_using_lama(f'./example/{ref_img}.png', f'{save_path}/mask_{ref_img}.png', lama_path, lama_config)
    pred_lama.save(f'{save_path}/lama_{ref_img}.png')

    # Refine the image by pasting the text
    img_lama = Image.open(f'{save_path}/lama_{ref_img}.png')
    img_ref = Image.open(f'./example/{ref_img}_ref.png')
    mask = Image.open(f'{save_path}/mask_{ref_img}.png')
    pred = infer(img_lama, img_ref, mask)
    pred.save(f'{save_path}/refine_{ref_img}.png')


if __name__ == '__main__':
    main()
