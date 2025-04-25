#!/usr/bin/env python
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.cluster import KMeans 
import itertools
import pulp
import cvxpy as cp
import pandas as pd 
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import pandas as pd 
from metrics.average_drop import AverageDrop
from metrics.average_increase import AverageIncrease


#----xu ly anh-----
def preprocess_image(image_path, device = "cpu"):
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
    }
    if method not in cam_map:
        raise ValueError(f"Unknown CAM method: {method}. Available: {list(cam_map.keys())}")
    CAMClass = cam_map[method]
    # Chỉ truyền model và target_layers
    return CAMClass(model=model, target_layers=[target_layer])

# --- 2. Lấy danh sách đường dẫn ảnh ---
def list_image_paths(image_dir):
    paths = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png")):
                paths.append(os.path.join(root, f))
    return sorted(paths)

# --- 3. Hàm chỉ lấy top-1 index cho mỗi ảnh ---
def predict_top1_indices(image_paths, model, device="cpu"):
    indices = []
    for p in image_paths:
        inp = preprocess_image(p, device)
        with torch.no_grad():
            out = model(inp)
            _, pred = out.max(1)
        indices.append(pred.item())
    return indices


# --- 4. Hàm chỉ tính grayscale CAM dựa vào top-1 index và đường dẫn ảnh ---
def compute_grayscale_cams(image_paths, cam, top1_indices, device="cpu"):
    cams = []
    for p, idx in zip(image_paths, top1_indices):
        inp = preprocess_image(p, device)
        # Không cần truyền use_cuda, CAM sẽ tự dùng device của model/input
        gcam_batch = cam(input_tensor=inp, targets=[ClassifierOutputTarget(idx)])
        cams.append(gcam_batch[0])
    return cams

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = models.resnet18(weights='IMAGENET1K_V1')# model = models.resnet18(weights='DEFAULT')
    model.eval().to(device)
    
    target_layer = model.layer4[-1]
    
    method = "gradcam"
    cam = get_cam(model, target_layer, method)
    
    image_dir = "datasets/imagenet"
    image_paths = list_image_paths(image_dir)
    image_paths = image_paths[:50]  # Chỉ lấy 10 ảnh đầu tiên để kiểm tra
    
    preprocessed_images = [preprocess_image(image_path, device) for image_path in image_paths]
    
    top1_indices = predict_top1_indices(image_paths, model, device)
    # print("Top-1 indices:", top1_indices)
    
    
    cam_maps = compute_grayscale_cams(image_paths, cam, top1_indices, device)
    
    average_drops = []
    increase_confidences = []
    
    # for img_tensor, saliency_map, target_cls in zip(
    #     preprocessed_images, cam_maps, top1_indices
    # ):
    #     drop, inc = calculate_metrics(
    #         model=model,
    #         image=img_tensor,
    #         saliency_map=saliency_map,
    #         target_class=target_cls,
    #         threshold=threashold
    # )
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
            device = device,
            apply_softmax=True,
        )
        # print("Average Drop:", drop)
        average_drops.append(drop)
        
        increase = average_increase(
            model=model,
            test_images=img_tensor,
            saliency_maps=saliency_map,
            class_idx=target_cls,
            device = device,
            apply_softmax=True,
        ) 
        # print("Increase Confidence:", increase)
        increase_confidences.append(increase)
    # average_drops = np.array(average_drops)
    # increase_confidences = np.array(increase_confidences)
    # print("Average Drop:", average_drops)
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

    excel_path = "results/gradcam.xlsx"
    results_df.to_excel(excel_path, index=False)

    print(f"Saved to path: {excel_path}")
    
    
    

    
