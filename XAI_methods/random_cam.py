import numpy as np
from .XAI_utils.cam_base import BaseCAM


class RandomCAM(BaseCAM):
    def __init__(self, model, target_layers, 
                 reshape_transform=None):
        super(
            RandomCAM,
            self).__init__(
            model,
            target_layers,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.random.uniform(-1, 1, size=(grads.shape[0], grads.shape[1]))