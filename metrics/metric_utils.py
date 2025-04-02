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
    if saliency_map.max() != 1 or saliency_map.min() != 0:
        print(f"Saliency map should have be normalized between 0 and 1. Current max value = {saliency_map.max()}, min value = {saliency_map.min()}")
        raise ValueError
    new_image = image * saliency_map
    return new_image 

#=========================================================================================================================================================# 

#1 Average Drop metric 
class AverageDrop(MetricBase):
    def __init__(self):
        super().__init__("Average_drop")
    def __call__(self, 
        model: nn.Module, 
        test_images: torch.Tensor, 
        saliency_maps: torch.Tensor, 
        class_idx: int | torch.Tensor, 
        device: str = "cpu", 
        apply_softmax: bool = True, 
        return_mean: bool = True,
        **kwargs, 
        ) -> torch.Tensor: 
        """
        The Average Drop refers to the maximum positive difference in the predictions made by the predictor using
        the input image and the prediction using the saliency map.
        Instead of giving to the model the original image, we give the saliency map as input and expect it to drop
        in performances if the saliency map doesn't contain relevant information.

        Args:
            model (torch.nn.Module): The model to be evaluated.
            test_images (torch.Tensor): The images to be tested, shape: (N, C, G, W)
            saliency_maps (torch.Tensor): The saliency maps to be evaluated, shape (N, C, H, W)
            class_idx (int): If int: the index of the class to be evaluated, the same for all the input images.
            if torch.Tensor: the index of the class to be evaluated for each input image. Shape: (N,)
        """
        test_images = test_images.to(device)
        # plt.imshow(test_images[0].permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()

        saliency_maps = saliency_maps.to(device)
        # plt.imshow(saliency_maps[0].permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()

        saliency_images = mix_image_with_saliency(test_images, saliency_maps)
        # plt.imshow(saliency_images[0].permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()

        test_preds = model(test_images)  # Shape: (N, num_classes)
        saliency_preds = model(saliency_images)  # Shape: (N, num_classes)

        if apply_softmax:
            test_preds = nn.functional.softmax(test_preds, dim=1)
            saliency_preds = nn.functional.softmax(saliency_preds, dim=1)

        #Select only the relevant class
        if isinstance(labels, int):
            test_preds = test_preds[:, labels]  # Shape: (N,)
            saliency_preds = saliency_preds[:, labels]  # Shape: (N,)
        elif isinstance(labels, torch.Tensor):
            test_preds = test_preds[torch.arange(test_preds.size(0)), labels]
            saliency_preds = saliency_preds[
                torch.arange(saliency_preds.size(0)), labels
            ]
        else:
            raise ValueError("class_idx should be either an int or a torch.Tensor")

        numerator = test_preds - saliency_preds
        numerator[numerator < 0] = 0

        denominator = test_preds

        res = torch.sum(numerator / denominator) * 100

        if return_mean:
            res = res.mean()

        return res.item()

# 2. increase in confidence 

from utils import AttributionMethod


class AverageIncrease(MetricBase):
    def __init__(self):
        super.__init__("AverageIncrease")
    
    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        saliency_maps: torch.Tensor,
        class_idx: int | torch.Tensor,
        attribution_method: AttributionMethod,
        device: torch.device | str = "cpu",
        apply_softmax: bool = True,
        return_mean: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        The number of times in the entire dataset that the model's confidence increased when providing only
        the saliency map as input.

        Args:

        model: torch.nn.Module
            The model to be evaluated.

        test_images: torch.Tensor
            The test images to be evaluated. Shape: (N, C, H, W)

        saliency_maps: torch.Tensor
            The saliency maps to be evaluated. Shape: (N, C, H, W)

        class_idx: int | torch.Tensor
            If int: the index of the class to be evaluated, the same for all the input images.
            if torch.Tensor: the index of the class to be evaluated for each input image. Shape (N,)
        """

        test_images = test_images.to(device)
        saliency_maps = saliency_maps.to(device)
        saliency_images = mix_image_with_saliency(test_images, saliency_maps)

        test_preds = model(test_images)  # Shape: (N, num_classes)
        saliency_preds = model(saliency_images)  # Shape: (N, num_classes)

        if apply_softmax:
            test_preds = nn.functional.softmax(test_preds, dim=1)
            saliency_preds = nn.functional.softmax(saliency_preds, dim=1)

        # Select only the relevant class
        if isinstance(class_idx, int):
            test_preds = test_preds[:, class_idx]  # Shape: (N,)
            saliency_preds = saliency_preds[:, class_idx]  # Shape: (N,)
        elif isinstance(class_idx, torch.Tensor):
            test_preds = test_preds[torch.arange(test_preds.size(0)), class_idx]
            saliency_preds = saliency_preds[
                torch.arange(saliency_preds.size(0)), class_idx
            ]
        else:
            raise ValueError("class_idx should be either an int or a torch.Tensor")

        numerator = test_preds - saliency_preds
        numerator[numerator > 0] = 0
        numerator[numerator < 0] = 1

        denominator = len(test_preds)  # N

        res = torch.sum(numerator / denominator) * 100

        if return_mean:
            res = res.mean()

        return res.item()

# 3. Deletion Curve 
from torcheval.metrics.aggregation.auc import AUC

class DeletionCurveAUC(MetricBase):
    def __init__(self, num_points: int = 30):
        super().__init__("deletion_curve_AUC")
        self.num_points = num_points

    def __call__(
        self,
        model: nn.Module,
        images: torch.Tensor,
        saliency_maps: torch.Tensor,
        labels: torch.Tensor,
        attribution_method: AttributionMethod,
        device: torch.device | str = "cpu",
        apply_softmax: bool = True,
        return_mean: bool = True,
        **kwargs,
    ):
        B, C, H, W = images.shape
        ins_range, insertion = self.deletion_curve(
            model, images, saliency_maps, labels, device, apply_softmax
        )
        res = torch.zeros(B)
        for i in range(B):
            insertion_auc = AUC()
            insertion_auc.update(ins_range[i], insertion[i])
            res[i] = insertion_auc.compute()

        if return_mean:
            res = res.mean()
        return res.item()
    
    def deletion_curve(
        self,
        model: nn.Module,
        images: torch.Tensor,
        saliency_maps: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device | str = "cpu",
        apply_softmax: bool = True,
        num_points: int = 30,
        ):
        """Generate the deletion curve as defined in https://arxiv.org/abs/1806.07421
        Args:
        model (nn.Module): The model to be evaluated
        image (torch.Tensor): The input image. Shape: (B, C, H, W)
        saliency_map (torch.Tensor): The saliency map. Shape: (B, C, H, W)
        device (torch.device | str, optional): The device to be used. Defaults to "cpu".
        labels (torch.Tensor): The labels of the input images. Shape: (B,)
        apply_softmax (bool, optional): Whether to apply softmax to the output. Defaults to True.
        """
        assert saliency_maps.shape[1] == 1, "Saliency map should be single channel"
        saliency_maps = saliency_maps.squeeze(1)
        
        B, C, H, W = images.shape
        num_pixels = H * W
        deletion_ranges = torch.zeros(B, self.num_points)
        deletion_values = torch.zeros(B, self.num_points)
        
        for b in range(B):
            image = images[b].unsqueeze(0)  # Shape: (1, C, H, W)
            # image = image.unsqueeze(0)  # Shape: (1, C, H, W)
            saliency_map = saliency_maps[b]  # Shape: (H, W)
            
            # Get indices sorted by saliency in descending order.
            sm_flatten = saliency_map.flatten()
            best_indices = sm_flatten.argsort().flip(0)  
            
            pixel_removed_perc = torch.linspace(0, 1, num_points)
            res = torch.zeros_like(pixel_removed_perc)
            
            for i, perc in enumerate(pixel_removed_perc):
                num_pixels_to_remove = int(num_pixels * perc)
                pixels_to_be_removed = best_indices[:num_pixels_to_remove]
                
                new_image = image.clone()
                new_image[0, :, pixels_to_be_removed // W, pixels_to_be_removed % W] = 0  # Remove the pixel by setting it to a constant value
                
                new_image = new_image.to(device)
                
                # Compute the prediction confidence on the class_idx
                with torch.no_grad():
                    preds = model(new_image)[0]
                    if apply_softmax:
                        preds = nn.functional.softmax(preds, dim=0)[labels[b]]
                    res[i] = preds

            deletion_ranges[b] = pixel_removed_perc
            deletion_values[b] = res

        return deletion_ranges, deletion_values


# 4. ROC_AUC

from sklearn.metrics import roc_auc_score


class ROC_AUC(MetricBase):
    def __init__(self):
        super().__init__("roc_auc")

    def __call__(
        self,
        saliency_maps: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # if "mask" not in kwargs:
        # raise ValueError("mask not provided in kwargs")
        # mask = kwargs["mask"]
        mask = mask.detach().cpu().numpy()
        attribution = saliency_maps.detach().cpu().numpy()

        if mask.shape != attribution.shape:
            raise ValueError(
                f"mask and attribution shape mismatch, {mask.shape} != {attribution.shape}"
            )

        if len(mask.shape) != 4:
            raise ValueError(
                f"mask and attribution should have 4 dimensions, actual shape: {mask.shape}"
            )

        if mask.shape[0] != 1 or mask.shape[1] != 1:
            raise ValueError(
                f"mask and attribution should have dimensions (1, 1, H, W), actual shape: {mask.shape}"
            )

        mask = mask.flatten()
        attribution = attribution.flatten()
        return roc_auc_score(mask, attribution)

# 5. Infidelity

# define a perturbation function for the input
def perturb_fn(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.tensor(
        np.random.normal(0, 0.003, inputs.shape), device=inputs.device
    ).float()
    return noise, inputs - noise


class Infidelity(BaseMetric):
    def __init__(self):
        super().__init__("infidelity")

    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        saliency_maps: torch.Tensor,
        class_idx: int | torch.Tensor,
        attribution_method: AttributionMethod,
        device: torch.device | str = "cpu",
        apply_softmax: bool = True,
        return_mean: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        saliency_maps = saliency_maps.expand(-1, 3, -1, -1).to(device)
        test_images = test_images.to(device)
        class_idx = class_idx.to(device)
        res = infidelity(
            model, perturb_fn, test_images, saliency_maps, target=class_idx
        )

        if return_mean:
            res = res.mean()

        return res.detach().cpu()
# 6 
from torchvision.transforms.functional import gaussian_blur
from torcheval.metrics.aggregation.auc import AUC


class InsertionCurveAUC(BaseMetric):
    def __init__(self):
        super().__init__("insertion_curve_AUC")

    def __call__(
        self,
        model: nn.Module,
        images: torch.Tensor,
        saliency_maps: torch.Tensor,
        labels: torch.Tensor,
        attribution_method: AttributionMethod,
        device: torch.device | str = "cpu",
        apply_softmax: bool = True,
        return_mean: bool = True,
        **kwargs,
    ):
        B, C, H, W = images.shape
        ins_range, insertion = insertion_curve(
            model, images, saliency_maps, labels, device, apply_softmax
        )
        res = torch.zeros(B)
        for i in range(B):
            insertion_auc = AUC()
            insertion_auc.update(ins_range[i], insertion[i])
            res[i] = insertion_auc.compute()

        if return_mean:
            res = res.mean()
        return res.item()


def insertion_curve(
    model: nn.Module,
    images: torch.Tensor,
    saliency_maps: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device | str = "cpu",
    apply_softmax: bool = True,
    num_points: int = 30,
):
    """Generate the insertion curve as defined in https://arxiv.org/abs/1806.07421

    Args:
        model (nn.Module): The model to be evaluated
        image (torch.Tensor): The input image. Shape: (C, H, W)
        saliency_map (torch.Tensor): The saliency map. Shape: (H, W)
    """
    assert saliency_maps.shape[1] == 1, "Saliency map should be single channel"
    saliency_maps = saliency_maps.squeeze(1)

    B, C, H, W = images.shape
    num_pixels = H * W

    insertion_ranges = torch.zeros(B, num_points)
    insertion_values = torch.zeros(B, num_points)

    for b in range(B):
        image = images[b].unsqueeze(0)  # Shape: (1, C, H, W)
        saliency_map = saliency_maps[b]  # Shape: (H, W)
        # Apply gaussian filter to the image
        # Values taken from https://github.com/eclique/RISE/blob/master/Evaluation.ipynb
        kernel_size = 11
        sigma = 5
        blurred_image = gaussian_blur(image, kernel_size, sigma)

        sm_flatten = saliency_map.flatten()
        best_indices = sm_flatten.argsort().flip(
            0
        )  # Indices of the saliency map sorted in descending order

        pixel_removed_perc = torch.linspace(0, 1, num_points)
        res = torch.zeros_like(pixel_removed_perc)

        # plt.figure(figsize=(20, 20))
        # for i, perc in tqdm(enumerate(pixel_removed_perc)):
        for i, perc in enumerate(pixel_removed_perc):
            # plt.subplot(6, 5, i + 1)
            num_pixels_to_remove = int(num_pixels * perc)

            pixels_to_be_removed = best_indices[:num_pixels_to_remove]

            new_image = blurred_image.clone()
            # new_image[0, :, pixels_to_be_removed // W, pixels_to_be_removed % W] = (
            #     0  # Remove the pixel by setting it to a constant value
            # )
            new_image[0, :, pixels_to_be_removed // W, pixels_to_be_removed % W] = (
                image[0, :, pixels_to_be_removed // W, pixels_to_be_removed % W]
            )

            new_image = new_image.to(device)

            # plt.imshow(new_image[0].permute(1, 2, 0).detach().cpu().numpy())
            # print(new_image[0].max(), new_image[0].min())

            # Compute the prediction confidence on the class_idx
            with torch.no_grad():
                preds = model(new_image)[0]
                if apply_softmax:
                    preds = nn.functional.softmax(preds, dim=0)[labels[b]]
                res[i] = preds

        insertion_ranges[b] = pixel_removed_perc
        insertion_values[b] = res

    return insertion_ranges, insertion_values


#6 Road 
from pytorch_grad_cam.metrics.road import (
    ROADCombined,
    ROADMostRelevantFirst,
    ROADLeastRelevantFirst,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget


class RoadCombined(BaseMetric):
    def __init__(self):
        super().__init__("road_combined")

    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        saliency_maps: torch.Tensor,
        class_idx: int | torch.Tensor,
        attribution_method: AttributionMethod,
        device: torch.device | str = "cpu",
        apply_softmax: bool = True,
        return_mean: bool = True,
        return_visualization: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        percentiles = [20, 40, 60, 80]
        road_combined = ROADCombined(percentiles=percentiles)
        targets = [ClassifierOutputSoftmaxTarget(i.item()) for i in class_idx]
        if len(saliency_maps.shape) == 4:
            saliency_maps = saliency_maps.squeeze(1)

        saliency_maps = saliency_maps.detach().cpu().numpy()
        scores = road_combined(test_images, saliency_maps, targets, model)

        if return_mean:
            scores = scores.mean()

        if not return_visualization:
            return scores

        # Calculate visualization
        visualization_results = []
        scores = []
        for imputer in [
            ROADMostRelevantFirst,
            ROADLeastRelevantFirst,
        ]:
            for perc in percentiles:
                score, visualizations = imputer(perc)(
                    test_images,
                    saliency_maps,
                    targets,
                    model,
                    return_visualization=True,
                )

                scores.append(score)

                # if imputer.__class__.__name__ not in visualization_results:
                #     visualization_results[imputer.__class__.__name__] = []

                visualization_results.append(visualizations[0].detach().cpu())

        return scores, visualization_results
 
 
# 7 Sensitive 
from captum.metrics import sensitivity_max
from utils import AttributionMethod


class Sensitivity(BaseMetric):
    def __init__(self):
        super().__init__("sensitivity")

    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        saliency_maps: torch.Tensor,
        class_idx: int | torch.Tensor,
        attribution_method: AttributionMethod,
        device: torch.device | str = "cpu",
        apply_softmax: bool = True,
        return_mean: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # **kwargs needs to contain baseline_dist and layer
        def attribution_wrapper(
            images: torch.Tensor, model, layer, targets, baseline_dist, **kwargs
        ):
            if type(images) is tuple and len(images) == 1:
                images = images[0]

            BATCH_SIZE = 1
            FINAL_SIZE = 10 // BATCH_SIZE
            ATTRIBUTION_SHAPE = None

            res = []
            for i in range(0, len(images), BATCH_SIZE):
                batch = images[i : i + BATCH_SIZE].requires_grad_().to(device)
                # Repeat targets
                batch_targets = torch.repeat_interleave(targets, BATCH_SIZE, dim=0)

                attribution_res = (
                    attribution_method.attribute(
                        batch, model, layer, batch_targets, baseline_dist
                    )
                    .detach()
                    .cpu()
                )
                ATTRIBUTION_SHAPE = attribution_res.shape

                # If any of the attributions is NaN, skip the batch
                if torch.isnan(attribution_res).any():
                    print("A saliency map is NaN, skipping batch")
                    del batch, batch_targets, attribution_res
                    torch.cuda.empty_cache()
                    continue
                res.append(attribution_res)

            if len(res) == 0:
                # Build a random very big tensor of the same shape of attributions
                res = [
                    torch.randn(ATTRIBUTION_SHAPE) * 9999999 for _ in range(FINAL_SIZE)
                ]

            if len(res) != FINAL_SIZE:
                remaining = FINAL_SIZE - len(res)
                res += [res[-1]] * remaining

            res = torch.cat(res, dim=0)

            return res

        # Set the **kwargs to contain model, layer, targets, baseline_dist
        kwargs["model"] = model
        kwargs["targets"] = class_idx

        sens = sensitivity_max(attribution_wrapper, test_images, **kwargs)

        if (sens > 100).any():
            return None

        if return_mean:
            sens = torch.mean(sens)

        return sens.detach().cpu()