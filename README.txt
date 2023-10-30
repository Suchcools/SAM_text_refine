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
