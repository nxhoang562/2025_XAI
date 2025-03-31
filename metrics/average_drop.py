import torch
import torch.nn as nn
from .utils import mix_image_and_saliency, BaseMetric
from utils import AttributionMethod


class AverageDrop(BaseMetric):
    def __init__(self):
        super().__init__("average_drop")
        pass

    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        saliency_maps: torch.Tensor,
        labels: int | torch.Tensor,
        attribution_method: AttributionMethod,
        device: torch.device | str = "cpu",
        apply_softmax: bool = True,
        return_mean: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        The Average Drop refers to the maximum positive difference in the predictions made by the predictor using
        the input image and the prediction using the saliency map.
        Instead of giving to the model the original image, we give the saliency map as input and expect it to drop
        in performances if the saliency map doesn't contain relevant information.

        Parameters:

        model: torch.nn.Module
            The model to be evaluated.

        test_images: torch.Tensor
            The test images to be evaluated. Shape: (N, C, H, W)

        saliency_maps: torch.Tensor
            The saliency maps to be evaluated. Shape: (N, C, H, W)

        class_idx: int | torch.Tensor
            If int: the index of the class to be evaluated, the same for all the input images.
            if torch.Tensor: the index of the class to be evaluated for each input image. Shape: (N,)
        """
        test_images = test_images.to(device)
        # plt.imshow(test_images[0].permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()

        saliency_maps = saliency_maps.to(device)
        # plt.imshow(saliency_maps[0].permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()

        saliency_images = mix_image_and_saliency(test_images, saliency_maps)
        # plt.imshow(saliency_images[0].permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()

        test_preds = model(test_images)  # Shape: (N, num_classes)
        saliency_preds = model(saliency_images)  # Shape: (N, num_classes)

        if apply_softmax:
            test_preds = nn.functional.softmax(test_preds, dim=1)
            saliency_preds = nn.functional.softmax(saliency_preds, dim=1)

        # Select only the relevant class
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
