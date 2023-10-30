# 文本识别+替换

使用DBNet++、SAM模型、Lama模型完成图像文字修复,并完成相应的接口设计
现在有两种图像，图像A是一个文本细节崩坏的图像，文本难以辨识。提供一个文本完好的参考图像A_ref，移除掉A中文本崩坏的区域，并将A_ref中文本完好的区域粘贴到A的对应位置。

实现的功能如下:
1. 通过DBNet++检测文本区域 (提供封装接口)
2. 通过 [SAM模型](https://github.com/facebookresearch/segment-anything) 获取更为精准的文本分割结果
3. 使用 [Lama模型](https://github.com/advimman/lama) 填充A中崩坏的文本区域
4. 将完好的文本粘贴到对应区域


mmocr和lama依赖库的安装，参考各自对应的```Readme```说明文档。

# 模型下载

[模型下载](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/dongzy6_mail2_sysu_edu_cn/EWqj4CWvXbBLjenjYwlaXL4BJOhrioRm4OGI2gE4wODUNg?e=du668G)

# 提供的封装接口

## 文本检测

使用以下接口实现文本检测:
```python
from mmocr.apis import TextDetInferencer
inferencer = TextDetInferencer(model='DBNetPP', weights=ckpt_path)
poly_list = inferencer(image_ref)['predictions'][0]['polygons']
for poly in poly_list:
    #poly: [x1,y1,x2,y2,...] 文字区域包围框的4个顶点
```

## 图像填充(Inpaint)

使用以下接口实现Inpaint:
```python
from lama.lama_interface import LamaInterface
from PIL import Image

lama = LamaInterface('big_lama/models/best.ckpt', 'big_lama/config.yaml')
img = Image.open('imgs/1.png')
mask = Image.open('mask/1.png') # 黑白mask，填充白色区域
pred = lama.infer(img, mask)
pred.save('1.png')
```

# 测试图像

用于测试结果的图像在`example/`文件夹中


主要文件有 
    segment_anything/SAMInterface
    run.py

在 run.py中配置以下路径既可运行 
    # Configurations
    ref_img = "1690952943765_1" # 文件ID
    save_path = "output" # 输出目录
    dbnetpp_path = "mmocr_text/ckpts/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth" # dbnetpp权重
    sam_path = "segment_anything/checkpoints/sam_vit_h_4b8939.pth" # SAM权重
    lama_path = "lama/big_lama/models/best.ckpt" # LAMA权重
    lama_config = "lama/big_lama/config.yaml" # LAMA配置

代码思路

1. 通过DBNet++检测文本区域,获取BOX
2. 根据获取BOX通过SAM获取更为精准的文本分割结果,获取掩码MASK文件
3. 使用步骤2 MASK文件用lama填充A中崩坏的文本区域,得到图片lama.png
4. 将ref文件通过mask覆盖掉lama.png完成填充
