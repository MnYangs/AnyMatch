import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
from onnxsim import simplify
import onnx

from src.edm.edm import EDM
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config



H, W = 480, 640


def read_gray(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (W, H))
    image = torch.from_numpy(image).float()
    image = image[None, None,] / 255.
    return image


def main():
    config = get_cfg_defaults()
    data_cfg_path = "../configs/data/scannet_test_1500.py"
    main_cfg_path = "../configs/edm/indoor/edm_base.py"
    config.merge_from_file(main_cfg_path)
    config.merge_from_file(data_cfg_path)
    
    config.EDM.DEPLOY = True  # export onnx model
    config.EDM.TEST_RES_H = H
    config.EDM.TEST_RES_W = W
    config.EDM.COARSE.TOPK = int(H / 8 * W / 8 * 0.35)
    
    _config = lower_config(config)
    matcher = EDM(config=_config["edm"])

    matcher.load_state_dict(torch.load(
        "../weights/edm_outdoor.ckpt")['state_dict'])
    matcher = matcher.eval()

    output_model_pth = f"./edm_w{config.EDM.TEST_RES_W}_h{config.EDM.TEST_RES_H}_topk{config.EDM.COARSE.TOPK}.onnx"
    
    with torch.no_grad():
        img0_pth = "../assets/scannet_sample_images/scene0707_00_15.jpg"
        img1_pth = "../assets/scannet_sample_images/scene0707_00_45.jpg"
        img0 = read_gray(img0_pth)
        img1 = read_gray(img1_pth)
        data = torch.concat([img0, img1], 1)  # NCHW
        print("input", data.shape)

        # TODO: dynamic axes
        torch.onnx.export(matcher, (data),
                          output_model_pth, verbose=True, input_names=['input'], output_names=['output'], opset_version=16)

    onnx_model = onnx.load(output_model_pth)

    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_model_pth)
    print('finished simplify onnx: ', output_model_pth)


if __name__ == "__main__":
    main()
