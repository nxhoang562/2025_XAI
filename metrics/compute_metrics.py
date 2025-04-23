import torch 
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt 

def calculate_metrics(
    model: nn.Module, 
    image: torch.Tensor,
    saliency_map: np.ndarray,
    target_class: int,
    threshold: float = 0.5,
) -> tuple[float, int]:
    """
    Tính toán các chỉ số Average Drop và Increase in Confidence
    https://arxiv.org/pdf/1910.01279
    
    """
    model.eval()
    with torch.no_grad():
        # Độ tin cậy (confidence) trên ảnh gốc
        logits_orig = model(image)  # [1, num_classes]
        probs_orig = torch.softmax(logits_orig, dim=1)
        Y_c = probs_orig[0, target_class].item()
        

        # Tạo ảnh masked từ saliency_map (giá trị > threshold giữ nguyên, còn lại là 0)
        saliency_map_tensor = torch.tensor(saliency_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        saliency_map_tensor = saliency_map_tensor
        if saliency_map_tensor.device != image.device:
            saliency_map_tensor = saliency_map_tensor.to(image.device)
        mask = (saliency_map_tensor > threshold).float()  
        masked_image = image * mask  # Chỉ giữ lại vùng saliency > threshold

        # Độ tin cậy trên bản đồ saliency
        logits_mask = model(masked_image)
        probs_mask = torch.softmax(logits_mask, dim=1)
        O_c = probs_mask[0, target_class].item()

        # Tính Average Drop
        average_drop = max(0, Y_c - O_c) / Y_c * 100  # càng nhỏ càng tốt 

        # Tính Increase in Confidence 
        increase_confidence = 1 if O_c > Y_c else 0  # Trả về 1 nếu O_c > Y_c càng cao càng tốt

    return average_drop, increase_confidence