import torch
from .build_sam import sam_model_registry
from .predictor import SamPredictor
import os
import numpy as np
import cv2


class SAMInterface:

    def __init__(self, model_type="vit_h", ckpt_path='segment_anything/checkpoints/sam_vit_h_4b8939.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam_model = self.load_sam_model(model_type, ckpt_path)
        self.predictor = SamPredictor(self.sam_model)

    def load_sam_model(self, model_type, ckpt_path):
        """
        加载 SAM 模型
        :param model_type: SAM 模型的类型
        :param ckpt_path: SAM 模型的检查点路径
        :return: 加载后的 SAM 模型
        """
        sam = sam_model_registry[model_type](
            checkpoint=ckpt_path).to(device=self.device)
        return sam

    @staticmethod
    def points_to_box(points):
        """
        将四个点的坐标表示为矩形框
        :param points: 包含四个点坐标的列表
        :return: 表示矩形框的 NumPy 数组 [x_min, y_min, x_max, y_max]
        """
        points = np.array(points).round().astype(np.int32).reshape(-1, 2)
        x_values = points[:, 0]
        y_values = points[:, 1]
        x_min = np.min(x_values)
        x_max = np.max(x_values)
        y_min = np.min(y_values)
        y_max = np.max(y_values)
        input_box = np.array([x_min, y_min, x_max, y_max])
        return input_box

    @torch.no_grad()
    def infer(self, poly_list: list, image_ref_path: str) -> np.array:
        """
        根据DBNet提供的粗文本框使用SAM模型细分文本框
        :param poly_list: DBNet接口输出的二维文字框列表
        :param image_ref: 具有清晰文本的参考图像（来自 mixpipe 的图像）
        :return: SAM 输出的合并后的掩码数组
        """

        box_list = [self.points_to_box(x) for x in poly_list]
        input_boxes = torch.tensor(box_list, device=self.device)

        image_ref = cv2.imread(image_ref_path)
        image_ref = cv2.cvtColor(image_ref, cv2.COLOR_BGR2RGB)
        
        self.predictor.set_image(image_ref)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            input_boxes, image_ref.shape[:2])
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # 将所有掩码相加以得到合并掩码
        merged_mask = torch.sum(masks, dim=0)

        # 转换为 NumPy 数组，并从第一个维度中移除尺寸为1的维度
        merged_mask = merged_mask.squeeze().cpu().numpy()
        merged_mask = np.array(merged_mask,np.uint8)
        return merged_mask


if __name__ == '__main__':
    sam_interface = SAMInterface(
        model_type='vit_h', ckpt_path='segment_anything/checkpoints/sam_vit_h_4b8939.pth')
    print(sam_interface)
