#!/usr/bin/env python
import os
import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd 
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM, AblationCAM, ShapleyCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# Import các CAM khác nếu bạn cần
# from pytorch_grad_cam import ScoreCAM, AblationCAM, EigenCAM
import numpy as np
from metrics.average_drop import AverageDrop
from metrics.average_increase import AverageIncrease

# ---- Xử lý ảnh ----
def preprocess_image(image_path, device="cpu"):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

def get_cam(model, target_layer, method: str = "gradcam"):
    method = method.lower()
    cam_map = {
        "gradcam": GradCAM,
        "gradcamplusplus": GradCAMPlusPlus,
        "layercam": LayerCAM,
        "scorecam": ScoreCAM,
        "ablationcam": AblationCAM,
        "shapleycam": ShapleyCAM,
    }
    if method not in cam_map:
        raise ValueError(f"Unknown CAM method: {method}. Available: {list(cam_map.keys())}")
    CAMClass = cam_map[method]
    return CAMClass(model=model, target_layers=[target_layer])

def list_image_paths(image_dir):
    paths = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png")):
                paths.append(os.path.join(root, f))
    return sorted(paths)

def predict_top1_indices(image_paths, model, device="cpu"):
    indices = []
    for p in image_paths:
        inp = preprocess_image(p, device)
        with torch.no_grad():
            out = model(inp)
            _, pred = out.max(1)
        indices.append(pred.item())
    return indices

def compute_grayscale_cams(image_paths, cam, top1_indices, device="cpu"):
    cams = []
    for p, idx in zip(image_paths, top1_indices):
        inp = preprocess_image(p, device)
        gcam_batch = cam(input_tensor=inp, targets=[ClassifierOutputTarget(idx)])
        cams.append(gcam_batch[0])
    return cams

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.eval().to(device)
    target_layer = model.layer4[-1]
    method = args.method.lower()

    cam = get_cam(model, target_layer, method)
    image_paths = list_image_paths(args.image_dir)
    image_paths = image_paths[:args.max_images]  # Số ảnh test tối đa

    preprocessed_images = [preprocess_image(image_path, device) for image_path in image_paths]
    top1_indices = predict_top1_indices(image_paths, model, device)
    cam_maps = compute_grayscale_cams(image_paths, cam, top1_indices, device)

    average_drops = []
    increase_confidences = []
    average_drop = AverageDrop()
    average_increase = AverageIncrease()
    for img_tensor, saliency_map, target_cls in zip(
        preprocessed_images, cam_maps, top1_indices
    ):
        drop = average_drop(
            model=model,
            test_images=img_tensor,
            saliency_maps=saliency_map,
            class_idx=target_cls,
            device=device,
            apply_softmax=True,
        )
        average_drops.append(drop)
        increase = average_increase(
            model=model,
            test_images=img_tensor,
            saliency_maps=saliency_map,
            class_idx=target_cls,
            device=device,
            apply_softmax=True,
        )
        increase_confidences.append(increase)
    results = pd.DataFrame({
        "image_path": image_paths,
        "top1_index": top1_indices,
        "average_drop": average_drops,
        "increase_confidence": increase_confidences,
    })
    avg_drop_mean = np.mean(average_drops)
    inc_conf_mean = np.mean(increase_confidences)
    average_row = pd.DataFrame([{
        "image_path": "AVERAGE",
        "top1_index": "",
        "average_drop": avg_drop_mean,
        "increase_confidence": inc_conf_mean
    }])
    results_df = pd.concat([results, average_row], ignore_index=True)

    excel_path = args.output_excel
    
    # Lưu ra nhiều sheet, sheet_name theo method
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a" if os.path.exists(excel_path) else "w") as writer:
        results_df.to_excel(writer, sheet_name=method.upper(), index=False)
    print(f"Saved to {excel_path}, sheet: {method.upper()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", "-m", type=str, default="gradcam", help="XAI method: gradcam, scorecam, ablationcam, eigencam,...")
    parser.add_argument("--image_dir", type=str, default="datasets/imagenet", help="Image folder")
    parser.add_argument("--output_excel", type=str, default="results/results_compare.xlsx", help="Output Excel file path")
    parser.add_argument("--max_images", type=int, default=50, help="Max images to process")
    args = parser.parse_args()
    main(args)
