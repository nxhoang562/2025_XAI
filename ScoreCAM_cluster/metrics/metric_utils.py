import torch 
import torch.nn as nn 
import numpy as np

class MetricBase: 
    def __init__(self, name: str):
        self.name = name 
    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        saliency_maps: torch.Tensor,
        class_idx: int | torch.Tensor,
        device: str = "cpu",
        apply_softmax: bool = True, 
        return_mean: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement __call__()") #Exception

def mix_image_with_saliency(
    image: torch.Tensor,
    saliency_map: torch.Tensor,
) -> torch.Tensor:
    """
    Mix the original image with the saliency map to create a new image: 
    Parameters: 
    - image (torch.Tensor): input image, shape (B, C, H, W)
    - saliency_map (torch.Tensor): saliency map, shape (B, C, H, W), element value is in (0,1)
    """  
    # if saliency_map.max() != 1 or saliency_map.min() != 0:
    #     print(f"Saliency map should have be normalized between 0 and 1. Current max value = {saliency_map.max()}, min value = {saliency_map.min()}")
    #     raise ValueError
    new_image = image * saliency_map
    return new_image 

#=========================================================================================================================================================# 