#!/usr/bin/env python
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
import math
import os 
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import preprocess_image
import numpy as np
import urllib.request
from metrics.average_drop import AverageDrop
from metrics.average_increase import AverageIncrease
from metrics.compute_metrics import calculate_metrics
import cv2
import torch.nn.functional as F


def preprocess_image(image_path, device = "cpu"):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

#===============================================================================
# Các hàm để trích xuất feature maps và nhóm chúng bằng KMeans clustering
#===============================================================================

def group_feature_maps(
    list_feature_maps: torch.Tensor | np.ndarray,
    number_of_clusters: int,
    ) -> list[tuple[int, ...]]:
    
    """
    Cluster channel maps (C, H, W) into num_clusters groups.
    Args:
        list_feature_maps (torch.Tensor or np.ndarray):
        Tensor or array of feature maps with shape (C, H, W) or (C, H*W).
        number_of_clusters (int): number of clusters to form.

    Returns:
        list[tuple[int, ...]]: Each tuple contains the channel indices in one cluster.
    """
    
    if not isinstance(number_of_clusters, int):
        raise ValueError("number of clusters must be an integer")
    
    # Convert to numpy array if torch.Tensor
    if isinstance(list_feature_maps, torch.Tensor):
        data = list_feature_maps.detach().cpu().numpy()
    else:
        data = list_feature_maps
    
    # Determine number of channels
    C = data.shape[0]
    if number_of_clusters < 1 or number_of_clusters > C:
        raise ValueError(
            f"number_of_clusters must be between 1 and {C}, got {number_of_clusters}"
        )
    
    # Flatten spatial dimensions (H, W) into feature vector
    flattened = data.reshape(C, -1)  # shape: (C, H*W)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0)
    labels = kmeans.fit_predict(flattened)
    
    # Build clusters of indices
    groups: list[tuple[int, ...]] = []
    for cluster_idx in range(number_of_clusters):
        idxs = tuple(int(i) for i in np.where(labels == cluster_idx)[0])
        groups.append(idxs)

    return groups



def get_feature_maps(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layers: nn.Module,
    ) -> list[torch.Tensor]:
    
    """
    Args:
        model (nn.Module): PyTorch model.
        input_tensor (torch.Tensor): Input tensor of shape (N, C, H, W).
        target_layers (list[nn.Module]): List of layers to extract feature maps from.
    """
    feature_maps = None
    
    def hook_fn(module, input, output):
        nonlocal feature_maps
        feature_maps = output.detach().cpu()
        return output
    # Register the forward hook on the target_layers
    hook = target_layers.register_forward_hook(hook_fn)
    _ = model(input_tensor)
    hook.remove()
    
    if feature_maps is None:
        raise ValueError("Cannot extract feature maps from target_layers.")
    
    return feature_maps
    
    
def group_last_conv_feature_maps(
    model: nn.Module,
    input: torch.Tensor,
    target_module: nn.Module,
    number_of_clusters: int,
    ) -> list:
    
    """ 
    Group the feature maps of the last convolutional layer of a model using KMeans clustering.
    Args:
        model (nn.Module): model to run the forward pass on
        input (torch.Tensor): input to the model 
        target_module (nn.Module): module to get the feature maps from
        number_of_clusters (int): number of clusters to form 
    Returns: Nhóm các chỉ số của các feature maps ví dụ [(0, 1), (2, 3), (4, 5)]; (0, 1) là cụm 1
    """
    feature_maps = None 
    
    def hook_fn(module, input, output):
        nonlocal feature_maps
        feature_maps = output.detach().cpu()
        return output
    # Register the forward hook on the target_module
    hook = target_module.register_forward_hook(hook_fn)
    _ = model(input)
    hook.remove()
    
    if feature_maps is None:
        raise ValueError("Cannot extract feature maps from target_module.")
    
    n_feature_maps = feature_maps.shape[1]  # C
    groups = group_feature_maps(feature_maps[0], number_of_clusters) #(1, C, H, W) => (C, H, W)
    
    return n_feature_maps, groups

#===============================================================================
# Các hàm để tính toán đóng góp của các nhóm feature maps
#===============================================================================

def do_masked_forward( model: nn.Module, 
                    input: torch.Tensor,
                    target_module: nn.Module,
                    mask_indices: list | torch.Tensor,
                    mask_value: float = 0.0,
                    ) -> torch.Tensor:
    
    """
    Run a forward pass through a model while masking specific channels in the output for a target module. 
    This is useful for analyzing the contribution of specific channels to the model's output.
    Args:
        model (nn.Module): model to run the forward pass on
        input (torch.Tensor): input to the model 
        mask_indices (list | torch.Tensor): indices of channels to mask
        mask_value (float, optional): Defaults to 0.0.
    """
    def hook_fn(module, input, output):
        # Mask the specified channels in the output
        device = output.device  # Lấy device của output (CPU hoặc GPU)
    
        if isinstance(mask_indices, list):
            idx = torch.tensor(mask_indices, dtype=torch.long, device=device)
        else:
            idx = mask_indices.to(dtype=torch.long, device=device)

        if output.dim() >= 2:  # (N, C, H, W) hoặc (N, C)
            # print(f"Idx: {idx}")
            output[:, idx, ...] = mask_value
    
    # Register the forward hook on the target_module
    hook = target_module.register_forward_hook(hook_fn)
    
    # Run the forward pass through the entire model
    out = model(input)
    
    # Remove the hook
    hook.remove()
    return out

def value_func(coalition: list[tuple[int, ...]],
                model: nn.Module,
                input: torch.Tensor,
                target_module: nn.Module,
                n_channels: int,
                target_class: int,
                mask_value: float = 0.0,
                ) -> float: 
    """
    Tính đóng góp của coalition dựa trên đầu ra của model. Các kênh không thuộc coalition sẽ bị mask bằng giá trị 0.0.
    Args:
        coalition (list):  Danh sách các nhãn nhóm (group labels) được giữ bật (unmasked). Ví dụ coalition = [(0, 1), (2, 3)]
        model (nn.Module): CNN model 
        input (torch.Tensor): ảnh đầu vào (B, C, H, W)
        target_module (nn.Module): lớp conv mà feature maps được trích xuất
        n_channels: số lượng kênh trong feature map
        target_class (int): class mục tiêu để tính toán đóng góp 
    """
    # 1) flatten toàn bộ channel trong coalition
    unmask = set()
    for group in coalition:
        unmask.update(group)
        
    # 2) danh sách các channel phải mask = những c ∉ unmask
    mask_indices = [c for c in range(n_channels) if c not in unmask]
    # print(f"Masking channels: {mask_indices}")
    
    # 3) chạy forward với hook
    with torch.no_grad():
        output = do_masked_forward(model, input, target_module, mask_indices, mask_value)
        probs = torch.nn.functional.softmax(output[0], dim=0)
    
    # top5 = torch.topk(probs, 5)
    # print("Top-5 class indices and probabilities:")
    # for idx, p in zip(top5.indices.tolist(), top5.values.tolist()):
    #     print(f"Class {idx}: {p:.4f}")
        
    # 4) trả về logit/score cho target_class
    target_output = float(probs[target_class].item())
    
    return target_output

def compute_group_contributions_shap(
    model: torch.nn.Module,
    input: torch.Tensor,
    target_module: torch.nn.Module,
    number_of_clusters: int,
    target_class: int,
    mask_value: float = 0.0
):
    """
    Tính exact Shapley values cho G nhóm feature-maps cuối cùng.

    Args:
        model: PyTorch model ở chế độ eval().
        input: Tensor shape (1, C, H, W), đã normalize.
        target_module: lớp convolution cuối cùng để trích feature-maps.
        number_of_clusters: số nhóm G.
        target_class: index class để tính Shapley value.
        mask_value: giá trị dùng để mask channels ngoài coalition.

    Returns:
        phi: np.ndarray shape (G,), Shapley values cho từng nhóm.
        group_contributions: list of dict {
            "group": tuple(channel indices...),
            "contribution": float(phi_i)
        }
    """
    # 1) Nhóm feature-maps
    n_channels, groups = group_last_conv_feature_maps(
        model, input, target_module, number_of_clusters
    )
    G = len(groups)
    idxs = list(range(G))

    # 2) Hàm kết quả v(S) = f(S) - f(empty)
    def f(coalition):
        return value_func(
            coalition, model, input, target_module,
            n_channels, target_class, mask_value
        )

    v_empty = f([])

    # 3) Precompute tất cả v(S) để cache
    v_cache = {}
    for r in range(0, G + 1):
        for S in itertools.combinations(idxs, r):
            coalition = [groups[i] for i in S]
            v_cache[S] = f(coalition) - v_empty

    # 4) Tính Shapley theo công thức tổ hợp
    phi = np.zeros(G, dtype=float)
    fact = math.factorial
    G_fact = fact(G)

    for i in idxs:
        for r in range(0, G):
            for S in itertools.combinations([j for j in idxs if j != i], r):
                S = tuple(S)
                S_union_i = tuple(sorted(S + (i,)))
                weight = fact(len(S)) * fact(G - len(S) - 1) / G_fact
                marginal = v_cache[S_union_i] - v_cache[S]
                phi[i] += weight * marginal

    # 5) Chuẩn bị output list
    group_contributions = [
        {"group": groups[i], "contribution": float(phi[i])}
        for i in range(G)
    ]

    return phi, group_contributions

#===============================================================================
# Các hàm để tính toán đại diện cho mỗi nhóm feature maps
#===============================================================================

def compute_group_representatives(
    feature_maps: torch.Tensor, 
    groups: list[tuple[int, ...]]) -> torch.Tensor:
    """
    Tính đại diện cho mỗi nhóm feature maps bằng cách lấy trung bình các kênh.
    
    Args:
        feature_maps (torch.Tensor): Tensor có shape (C, H, W) với C là số kênh.
        groups (list[tuple[int, ...]]): Danh sách các tuple chứa chỉ số các kênh cho mỗi nhóm.
    
    Returns:
        torch.Tensor: Tensor có shape (G, H, W), với G là số nhóm.
    """
    representatives = []
    for group in groups:
        # Lấy các feature map theo chỉ số trong group và tính trung bình theo kênh
        group_maps = feature_maps[list(group)]  # shape: (len(group), H, W)
        rep = group_maps.mean(dim=0)  # shape: (H, W)
        representatives.append(rep)
    return torch.stack(representatives, dim=0)  # shape: (G, H, W)

def softmax(contributions):
    contributions = np.array(contributions)
    exp_values = np.exp(contributions - np.max(contributions))  # ổn định số học
    return exp_values / np.sum(exp_values)

def compute_weighted_group_representatives(
    feature_maps: torch.Tensor, 
    groups: list[tuple[int, ...]],
    contributions: list[float]
) -> torch.Tensor:
    """
    Tính weighted representatives cho mỗi nhóm bằng cách nhân đại diện (tính theo trung bình)
    với trọng số contribution của nhóm đó.
    
    Args:
        feature_maps (torch.Tensor): Tensor có shape (C, H, W) với C là số kênh.
        groups (list[tuple[int, ...]]): Danh sách các tuple chứa chỉ số các kênh cho mỗi nhóm.
        contributions (list[float]): Danh sách contribution tương ứng với mỗi nhóm.
        
    Returns:
        torch.Tensor: Tensor có shape (G, H, W), với G là số nhóm.
    """
    if len(groups) != len(contributions):
        raise ValueError("Độ dài của groups và contributions phải bằng nhau.")
        
    
    # Normalize contributions to sum to 1
    contributions = torch.tensor(contributions, dtype=torch.float32)
    contributions /= contributions.sum()
    # contributions_norm = softmax(contributions)
    
    weighted_reps = []
    for group, weight in zip(groups, contributions):
        group_maps = feature_maps[list(group)]    # shape: (len(group), H, W) -> lấy feature maps theo chỉ số trong group
        rep = group_maps.mean(dim = 0) # tính trung bình theo kênh , shape (H , W)
        weighted_rep = weight * rep                 # tính weighted representative
        weighted_reps.append(weighted_rep)
    return torch.stack(weighted_reps, dim=0)  

#===============================================================================
# Hàm chính để tính toán saliency map
#===============================================================================

def compute_saliency_map(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_module: torch.nn.Module,
    number_of_clusters: int,
    target_class: int,
    mask_value: float = 0.0
) -> torch.Tensor:
    """
    Tạo saliency map cho một ảnh.
    Args:
        model: PyTorch model ở chế độ eval().
        input_tensor: Tensor shape (1, 3, 224, 224), đã normalize.
        target_module: lớp conv cuối cùng.
        number_of_clusters: số cụm.
        target_class: index class để tính contributions.
        mask_value: giá trị mask (mặc định 0).
    Returns:
        saliency_map: numpy array shape (224, 224).
    """
    # 1) Phân cụm và lấy groups
    _, groups = group_last_conv_feature_maps(
        model, input_tensor, target_module, number_of_clusters
    )
    # 2) Tính contributions
    phi, group_contributions = compute_group_contributions_shap(
        model,
        input=input_tensor,
        target_module=target_module,
        number_of_clusters=number_of_clusters,
        target_class=target_class,
        mask_value=mask_value
    )
    # 3) Trích feature maps đầy đủ
    fmap_batch = get_feature_maps(model, input_tensor, target_module)
    full_feature_maps = fmap_batch[0]  # (C, H, W)
    # 4) Tính weighted representatives
    contrib_values = [rec['contribution'] for rec in group_contributions]
    weighted_reps = compute_weighted_group_representatives(
        full_feature_maps, groups, contrib_values
    )  # (G, H, W)
    summed = torch.sum(weighted_reps, dim=0)  # (H, W)
    # 5) ReLU và normalize
    saliency = F.relu(summed).cpu()
    saliency_np = saliency.numpy()
    saliency_norm = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min() + 1e-8)
    # 6) Resize về 224×224
    sal_map = cv2.resize(saliency_norm, (224, 224), interpolation=cv2.INTER_LINEAR)
    return sal_map# shape: (G, H, W)

#===============================================================================
# Hàm để lấy danh sách đường dẫn ảnh trong thư mục
#===============================================================================

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



if __name__ == "__main__":
    # --- 1. Load model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.eval().to(device)
    target_layer = model.layer4[-1]

    # --- 2. Lấy danh sách đường dẫn ảnh ---
    image_dir = "datasets/imagenet"
    image_paths = list_image_paths(image_dir)[:50]

    # --- 3. Hàm chỉ lấy top-1 index cho mỗi ảnh ---
    top1_idxs = predict_top1_indices(image_paths, model, device)

    average_drop = AverageDrop()
    average_increase = AverageIncrease()

    k_values = [10,11,13,15]
    excel_path = " /home/infres/xnguyen-24/XAI/results/10-img_shap.xlsx"
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)

    for c in k_values:
        print(f"\n=== Processing with number of cluster = {c} ===")
        average_drops = []
        increase_confidences = []

        # Tính saliency và metrics
        for idx, (path, target_cls) in enumerate(zip(image_paths, top1_idxs), 1):
            print(f"Processing image {idx}/{len(image_paths)}: {path}")
            img_tensor = preprocess_image(path, device)
            saliency_map = compute_saliency_map(
                model, img_tensor, target_layer,
                number_of_clusters=c, target_class=target_cls
            )

            drop = average_drop(
                model=model,
                test_images=img_tensor,
                saliency_maps=saliency_map,
                class_idx=target_cls,
                device=device,
                apply_softmax=True,
            )
            average_drops.append(drop)

            inc = average_increase(
                model=model,
                test_images=img_tensor,
                saliency_maps=saliency_map,
                class_idx=target_cls,
                device=device,
                apply_softmax=True,
            )
            increase_confidences.append(inc)

        # Tạo DataFrame kết quả và thêm dòng trung bình
        results_df = pd.DataFrame({
            "image_path": image_paths,
            "top1_index": top1_idxs,
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
        results_df = pd.concat([results_df, average_row], ignore_index=True)

        # Ghi vào Excel: nếu file đã tồn tại thì append + replace sheet, không thì tạo mới
        sheet_name = f"sum_norm_num_clusters_{c}"
        file_exists = os.path.exists(excel_path)
        writer_kwargs = {
            "engine": "openpyxl",
            "mode": "a" if file_exists else "w"
        }
        if file_exists:
            writer_kwargs["if_sheet_exists"] = "replace"

        with pd.ExcelWriter(excel_path, **writer_kwargs) as writer:
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"→ Saved sheet `{sheet_name}` {'(appended)' if file_exists else '(created)'} in {excel_path}")

