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
import os 
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import preprocess_image
import numpy as np
import json
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
    target_module: nn.Module,
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
    # Register the forward hook on the target_module
    hook = target_module.register_forward_hook(hook_fn)
    _ = model(input_tensor)
    hook.remove()
    
    if feature_maps is None:
        raise ValueError("Cannot extract feature maps from target_module.")
    
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

def compute_group_contributions(
    model: nn.Module,
    input: torch.Tensor,
    target_module: nn.Module,
    number_of_clusters: int,
    target_class: int,
    mask_value: float = 0.0,
):
    """
    Solve for the pre-nucleolus allocation x ∈ R^G over G groups of feature‐maps.

    Returns:
        x_opt: ndarray of shape (G,)
        eps_opt: float
        group_contributions: list of dict {group: tuple, contribution: float}
    """
    # 1) Nhóm feature maps
    n_channels, groups = group_last_conv_feature_maps(
        model, input, target_module, number_of_clusters
    )
    G = len(groups)

    # --- Helper: tính f(coalition) ---
    def f(coalition: list[tuple[int,...]]) -> float:
        """
        coalition: list các group (mỗi group là tuple các channel indices)
        Trả về xác suất softmax của target_class khi chỉ mở những group này.
        """
        return value_func(
            coalition, model, input, target_module,
            n_channels, target_class, mask_value
        )

    # 2) Tính baseline và v(S) đã normalize
    f_empty = f([])        # mask hết mọi group → baseline
    full = tuple(range(G)) # (0,1,2,…,G-1)

    # v là dict từ tuple of indices -> giá trị v(S)
    v: dict[tuple[int,...], float] = {}

    # 2.1) full coalition
    v[full] = f(groups) - f_empty

    # 2.2) mọi coalition con
    for r in range(1, G):
        for combo in itertools.combinations(range(G), r):
            coalition = [groups[i] for i in combo]
            raw = f(coalition) - f_empty
            v[combo] = max(raw, 0.0)  # floor tại 0

    # 3) Xây LP pre-nucleolus
    x   = cp.Variable(G, nonneg=True)
    eps = cp.Variable(nonneg=True)

    # Efficiency
    cons = [cp.sum(x) == v[full]]
    # Core constraints
    for S, val in v.items():
        # S tuple rỗng chưa được thêm vì r chạy từ 1..G-1
        cons.append(cp.sum(x[list(S)]) >= val - eps)

    prob = cp.Problem(cp.Minimize(eps), cons)

    # 4) Thử solver ECOS, nếu không cài ECOS thì fallback sang SCS
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except cp.error.SolverError:
        prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"LP vô nghiệm hoặc thất bại: {prob.status}")

    x_opt = x.value
    eps_opt = eps.value
    group_contributions = [
        {"group": groups[i], "contribution": float(x_opt[i])}
        for i in range(G)
    ]

    return x_opt, eps_opt, group_contributions



# def compute_group_contributions(
#     model: nn.Module,
#     input: torch.Tensor,
#     target_module: nn.Module,
#     number_of_clusters: int,
#     target_class: int ,
#     mask_value: float = 0.0,
# ):
#     """
#     Solve for the nucleolus allocation x ∈ R^G over G groups of feature‐maps.
    
#     Args: 
#         model: DNN model
#         input: Ảnh đầu vào 
#         target_module: Lớp tích chập extract feature maps
#         number_of_clusters: số lượng nhóm kênh 
#         target_class: class mục tiêu để tính toán đóng góp
#         mask_value: giá trị để mask các kênh không thuộc coalition, default = 0 
#     Returns:
#         x_opt: ndarray of shape (G,)
#         epsilon_opt: the minimized maximum excess

#     """
#     # 1) group feature maps
#     n_channels , groups = group_last_conv_feature_maps(model, input, target_module, number_of_clusters) #e.g: groups = [(0, 1), (2, 3), (4, 5)]
#     # print("Groups:", groups)
#     # Số lượng kênh 
#     G = len(groups)

#     # 2) Tính v(S) cho mỗi coalition S (tập tuple group indices) ⊂ {0,…,G−1}
#     coalitions = []
#     for r in range(1, G+1):
#         for combo in itertools.combinations(range(G), r):
#             coalition = [groups[i] for i in combo]
#             coalitions.append((combo, coalition))
#     # Bao gồm full coalition
#     full_idx = tuple(range(G))
    

#     # Tính v cho từng coalition
#     v: dict[tuple[int, ...], float] = {} # {"(0,1)": 0.5, "(2,3)": 0.3}
#     v[full_idx] = value_func(groups, model, input, target_module, n_channels, target_class , mask_value)
#     print("v(full_idx):", v[full_idx])
#     for idxs, coalition in coalitions:
#         if idxs != full_idx:
#             v[idxs] = value_func(coalition, model, input, target_module, n_channels, target_class, mask_value)
    
#     records = [
#         {"coalition": str(coal), "value": val}
#         for coal, val in v.items()
#     ]
#     df = pd.DataFrame(records)
#     # csv_path = "results/coalition_values.csv"
#     # df.to_csv(csv_path, index=False)
#     # print(f"Coalition values saved to {csv_path}")
            

#    # 3) Khởi tạo biến CVXPY
#     x   = cp.Variable(G, nonneg=True)   # x_i >= 0
#     eps = cp.Variable(nonneg=True)      # ε >= 0
#     # cons = [cp.sum(x) == v[full_idx], x >= 0, eps >= 0]
#     cons = [cp.sum(x) == v[full_idx]]
#     for S, val in v.items():
#         if S != full_idx:
#             cons.append(cp.sum(x[list(S)]) >= val - eps)

#     prob = cp.Problem(cp.Minimize(eps), cons)
#     # prob.solve(solver=cp.SCS)
#     prob.solve(solver=cp.GLPK, glpk={'msg_lev': 'GLP_MSG_OFF'})
#     if prob.status not in (cp.OPTIMAL, "optimal"):
#         # raise RuntimeError(f"LP did not solve: {prob.status}")
#         raise RuntimeError(f"Pre-nucleolus LP infeasible or failed: {prob.status}")
    
#     group_contributions = [ {"group": groups[i], "contribution": float(x.value[i])} for i in range(G)]
    
#     return x.value, eps.value,  group_contributions


# def compute_group_contributions(
#     model: nn.Module,
#     input: torch.Tensor,
#     target_module: nn.Module,
#     number_of_clusters: int,
#     target_class: int,
#     mask_value: float = 0.0,
# ):
#     """
#     Solve for the nucleolus allocation x ∈ R^G over G groups of feature‐maps,
#     với các ràng buộc:
#       • sum_i x_i = v(N)
#       • x_i >= v({i})  (individual rationality)
#       • for every S⊂N, sum_{i∈S} x_i >= v(S) - eps  (lexicographic‐LP slack)
#     """

#     # 1) Nhóm feature‐maps
#     n_channels, groups = group_last_conv_feature_maps(
#         model, input, target_module, number_of_clusters
#     )
#     G = len(groups)
#     full_idx = tuple(range(G))

#     # 2) Tính v(S) cho mọi coalition
#     coalitions = []
#     for r in range(1, G + 1):
#         for combo in itertools.combinations(range(G), r):
#             coalition = [groups[i] for i in combo]
#             coalitions.append((combo, coalition))

#     # Dùng dict để chứa giá trị
#     v: dict[tuple[int, ...], float] = {}
#     # Grand coalition
#     v[full_idx] = value_func(
#         groups, model, input, target_module, n_channels, target_class, mask_value
#     )
#     # Các coalition khác (bao gồm cả singletons)
#     for idxs, coalition in coalitions:
#         if idxs != full_idx:
#             v[idxs] = value_func(
#                 coalition, model, input, target_module, n_channels, target_class, mask_value
#             )

#     # 3) Tạo biến CVXPY
#     x = cp.Variable(G)
#     eps = cp.Variable()

#     # 4) Khởi tạo constraints
#     constraints = [
#         cp.sum(x) == v[full_idx],   # efficiency
#         eps >= 0                    # slack không âm
#     ]

#     # 4.1) Lexicographic‐LP constraints cho mọi S⊂N, S≠N
#     for S, val in v.items():
#         if S != full_idx:
#             constraints.append(cp.sum(x[list(S)]) >= val - eps)

#     # 4.2) Individual rationality: x[i] >= v({i})
#     # Vì v đã chứa luôn giá trị của mọi singleton coalition (i.e. S=(i,))
#     for i in range(G):
#         constraints.append(x[i] >= v[(i,)])

#     # 5) Giải LP
#     prob = cp.Problem(cp.Minimize(eps), constraints)
#     prob.solve(solver=cp.SCS)

#     if prob.status not in (cp.OPTIMAL, 'optimal'):
#         raise RuntimeError(f"LP did not solve: {prob.status}")

#     # 6) Trả kết quả
#     group_contributions = [
#         {"group": groups[i], "contribution": float(x.value[i])}
#         for i in range(G)
#     ]
#     return x.value, eps.value, group_contributions



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
    _, _, group_contributions = compute_group_contributions(
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
    image_paths = list_image_paths(image_dir)[:100]

    # --- 3. Hàm chỉ lấy top-1 index cho mỗi ảnh ---
    top1_idxs = predict_top1_indices(image_paths, model, device)

    average_drop = AverageDrop()
    average_increase = AverageIncrease()

    k_values = [8,9,10,11,15,20]
    excel_path = "results/100img_baseline_pre_nucleolus.xlsx"
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

    # print(f"Appended sheet 'our_results' into {excel_path}")
        
    
    # for idx, (path, target_cls) in enumerate(zip(image_paths , top1_idxs), 1):  
    #     print(f"Processing image {idx}/{len(image_paths )}: {path}")
    #     # Preprocess & compute saliency
    #     img_tensor = preprocess_image(path, device)
    #     saliency_map = compute_saliency_map(
    #         model, img_tensor, target_layer,
    #         number_of_clusters=15, target_class=target_cls
    #     )
    #     results.append((path, saliency_map))
        
    #     drop = average_drop(
    #         model=model,
    #         test_images=img_tensor,
    #         saliency_maps=saliency_map,
    #         class_idx=target_cls,
    #         device = device,
    #         apply_softmax=True,
    #     )
    #     average_drops.append(drop)
        
    #     increase = average_increase(
    #         model=model,
    #         test_images=img_tensor,
    #         saliency_maps=saliency_map,
    #         class_idx=target_cls,
    #         device = device,
    #         apply_softmax=True,
    #     ) 
    #     # print("Increase Confidence:", increase)
    #     increase_confidences.append(increase)
    
    # # print("length of image_path", len(image_paths))
    # # print("length of top1_index", len(top1_idxs))
    # # print("length of average_drops", len(average_drops))
    # # print("length of increase_confidences", len(increase_confidences))
        
    
    # results = pd.DataFrame({
    #     "image_path":  image_paths ,
    #     "top1_index":  top1_idxs,
    #     "average_drop": average_drops,
    #     "increase_confidence": increase_confidences,
    # })
    # avg_drop_mean = np.mean(average_drops)
    # inc_conf_mean = np.mean(increase_confidences)
    
    # average_row = pd.DataFrame([{
    # "image_path": "AVERAGE",
    # "top1_index": "",
    # "average_drop": avg_drop_mean,
    # "increase_confidence": inc_conf_mean
    # }])
    
    # results_df = pd.concat([results, average_row], ignore_index=True)

    # excel_path = "results/results.xlsx"
    # with pd.ExcelWriter(excel_path,
    #                 engine='openpyxl',
    #                 mode='a',                  
    #                 if_sheet_exists='replace'  # hoặc 'new' nếu bạn muốn giữ sheet cũ và tạo sheet mới có tên trùng lặp
    #                ) as writer:
    #     results_df.to_excel(writer,
    #                     sheet_name='k_15',  # đổi tên sheet tuỳ bạn
    #                     index=False)
    # print(f"Appended sheet 'our_results' into {excel_path}")
    
    
    
        

    
    
    