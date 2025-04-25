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
from pytorch_grad_cam.utils.image import preprocess_image
import numpy as np
import json
import urllib.request
from metrics.average_drop import AverageDrop
from metrics.average_increase import AverageIncrease
from metrics.compute_metrics import calculate_metrics

# co the su dung global average pooling cho moi feature map 
# hien tai dang chuyen thanh vector 1 chieu 

def group_feature_maps(
    list_feature_maps: torch.Tensor | np.ndarray,
    number_of_clusters: int,
    ) -> list[tuple[int, ...]]:
    
    """
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
    target_class: int ,
    mask_value: float = 0.0,
):
    """
    Solve for the nucleolus allocation x ∈ R^G over G groups of feature‐maps.
    
    Args: 
        model: DNN model
        input: Ảnh đầu vào 
        target_module: Lớp tích chập extract feature maps
        number_of_clusters: số lượng nhóm kênh 
        target_class: class mục tiêu để tính toán đóng góp
        mask_value: giá trị để mask các kênh không thuộc coalition, default = 0 
    Returns:
      x_opt: ndarray of shape (G,)
      epsilon_opt: the minimized maximum excess

    """
    # 1) group feature maps
    n_channels , groups = group_last_conv_feature_maps(model, input, target_module, number_of_clusters) #e.g: groups = [(0, 1), (2, 3), (4, 5)]
    print("Groups:", groups)
    # Số lượng kênh 
    G = len(groups)

    # 2) Tính v(S) cho mỗi coalition S (tập tuple group indices) ⊂ {0,…,G−1}
    coalitions = []
    for r in range(1, G+1):
        for combo in itertools.combinations(range(G), r):
            coalition = [groups[i] for i in combo]
            coalitions.append((combo, coalition))
    # Bao gồm full coalition
    full_idx = tuple(range(G))
    

    # Tính v cho từng coalition
    v: dict[tuple[int, ...], float] = {} # {"(0,1)": 0.5, "(2,3)": 0.3}
    v[full_idx] = value_func(groups, model, input, target_module, n_channels, target_class , mask_value)
    print("v(full_idx):", v[full_idx])
    for idxs, coalition in coalitions:
        if idxs != full_idx:
            v[idxs] = value_func(coalition, model, input, target_module, n_channels, target_class, mask_value)
    
    records = [
        {"coalition": str(coal), "value": val}
        for coal, val in v.items()
    ]
    df = pd.DataFrame(records)
    csv_path = "/home/infres/xnguyen-24/XAI/results/coalition_values.csv"
    df.to_csv(csv_path, index=False)
    print(f"Coalition values saved to {csv_path}")
            

   # 3) Khởi tạo biến CVXPY
    x = cp.Variable(G)
    eps = cp.Variable()
    cons = [cp.sum(x) == v[full_idx], x >= 0, eps >= 0]
    for S, val in v.items():
        if S != full_idx:
            cons.append(cp.sum(x[list(S)]) >= val - eps)

    prob = cp.Problem(cp.Minimize(eps), cons)
    prob.solve(solver=cp.SCS)
    if prob.status not in (cp.OPTIMAL, "optimal"):
        raise RuntimeError(f"LP did not solve: {prob.status}")
    
    group_contributions = [ {"group": groups[i], "contribution": float(x.value[i])} for i in range(G)]
    
    return x.value, eps.value,  group_contributions


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