# pip install importlib_resources scikit-learn openpyxl

import os
import sys
import torch
import torch.nn.functional as F
import torchvision.models as models
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ScoreCAM_cluster.utils import load_image, apply_transforms, basic_visualize, list_image_paths, predict_top1_indices, preprocess_image
from ScoreCAM_cluster.cam.clusterscorecam import ClusterScoreCAM
from metrics.metric_utils import AverageDrop, AverageIncrease


def test_single_image(
    model, model_dict, img_path, save_prefix, num_clusters=5, device=None
):
    """
    Chạy ClusterScoreCAM và tính metric AverageDrop, AverageIncrease cho 1 ảnh.
    Trả về tuple (drop, increase).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # khởi tạo CAM và metric
    cam = ClusterScoreCAM(model_dict, num_clusters=num_clusters)
    avg_drop = AverageDrop()
    avg_inc = AverageIncrease()

    # load & preprocess
    img = load_image(img_path)
    inp = apply_transforms(img).to(device)  # (1,3,224,224)

    # dự đoán nhãn top-1
    with torch.no_grad():
        logits = model(inp)
        target_cls = logits.argmax(dim=1).item()

    # tính saliency map
    sal_map = cam(inp, class_idx=target_cls)
    sal_map = sal_map.cpu().squeeze(0)  # (1,H,W)

    # lưu ảnh heatmap
    basic_visualize(
        inp.cpu().squeeze(0),
        sal_map,
        save_path=f"{save_prefix}_clusters{num_clusters}.png"
    )

    # mở rộng kênh cho metric (1,3,H,W)
    sal3 = sal_map.unsqueeze(0).repeat(1, inp.size(1), 1, 1)

    # tính metrics
    drop_val = avg_drop(
        model=model,
        test_images=inp,
        saliency_maps=sal3,
        class_idx=target_cls,
        device=device,
        apply_softmax=True,
        return_mean=True
    )
    inc_val = avg_inc(
        model=model,
        test_images=inp,
        saliency_maps=sal3,
        class_idx=target_cls,
        device=device,
        apply_softmax=True,
        return_mean=True
    )
    return drop_val, inc_val


def batch_test(
    model, model_dict, image_dir, excel_path, k_values, top_n=100
):
    """
    Test ClusterScoreCAM trên nhiều ảnh và nhiều giá trị K, lưu kết quả vào Excel theo sheet.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    # target_layer không cần trực tiếp, dùng model_dict

    # --- danh sách ảnh và nhãn top1 ---
    image_paths = list_image_paths(image_dir)[:top_n]
    top1_idxs = predict_top1_indices(image_paths, model, device)

    # chuẩn bị file Excel
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    for c in k_values:
        print(f"\n=== Testing with K={c} ===")
        drops, incs = [], []
        for idx, (path, cls) in enumerate(zip(image_paths, top1_idxs), 1):
            print(f"[{idx}/{len(image_paths)}] {os.path.basename(path)} -> class {cls}")
            # preprocess_image sẽ load + transform + unsqueeze
            img_tensor = preprocess_image(path, device)
            # compute saliency map
            sal_map = ClusterScoreCAM(model_dict, num_clusters=c)(
                img_tensor, class_idx=cls
            )
            sal_map = sal_map.cpu().squeeze(0)
            sal3 = sal_map.unsqueeze(0).repeat(1, img_tensor.size(1), 1, 1)
            # metrics
            drop = AverageDrop()(model, img_tensor, sal3, cls, device, True)
            inc = AverageIncrease()(model, img_tensor, sal3, cls, device, True)
            drops.append(drop)
            incs.append(inc)
        # tạo DataFrame
        df = pd.DataFrame({
            "image_path": image_paths,
            "top1_index": top1_idxs,
            "average_drop": drops,
            "increase_confidence": incs,
        })
        avg_row = pd.DataFrame([{  
            "image_path": "AVERAGE",  
            "top1_index": "",  
            "average_drop": np.mean(drops),
            "increase_confidence": np.mean(incs)
        }])
        df = pd.concat([df, avg_row], ignore_index=True)

        # ghi Excel
        sheet = f"sum_norm_num_clusters_{c}"
        mode = "a" if os.path.exists(excel_path) else "w"
        with pd.ExcelWriter(
            excel_path, engine="openpyxl", mode=mode,
            if_sheet_exists="replace" if mode=="a" else None
        ) as writer:
            df.to_excel(writer, sheet_name=sheet, index=False)
        print(f"→ Saved sheet `{sheet}` in {excel_path}")


if __name__ == "__main__":
    # ví dụ chạy batch test
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    model_dict = {
        'type': 'resnet18',
        'arch': model,
        'layer_name': 'layer4',
        'input_size': (224,224)
    }
    image_dir = "datasets/imagenet"
    excel_path = "results/100img_baseline_pre_nucleolus.xlsx"
    k_vals = [8,9,10,11,15,20]
    batch_test(model, model_dict, image_dir, excel_path, k_vals, top_n=100)
