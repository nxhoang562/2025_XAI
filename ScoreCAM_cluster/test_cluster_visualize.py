# pip install importlib_resources scikit-learn

import torch
import torch.nn.functional as F
import torchvision.models as models

from utils import load_image, apply_transforms, basic_visualize
from cam.clusterscorecam import ClusterScoreCAM

def test_cluster_scorecam(model, model_dict, img_path, save_prefix, num_clusters=5):
    cam = ClusterScoreCAM(model_dict, num_clusters=num_clusters)
    
    # load & preprocess
    img = load_image(img_path)
    inp = apply_transforms(img)   # (1,3,224,224)
    if torch.cuda.is_available():
        inp = inp.cuda()
    
    # chạy CAM
    heatmap = cam(inp)                         # (1,1,224,224)
    heatmap = heatmap.cpu().squeeze(0)  # (1, 224,224)
    
    # visualize
    basic_visualize(
        inp.cpu().detach().squeeze(0), 
        heatmap, 
        save_path=f"{save_prefix}_clusters{num_clusters}.png"
    )


if __name__ == "__main__":
    img_file = "images/ILSVRC2012_val_00002193.JPEG"

    # # AlexNet
    # alexnet = models.alexnet(pretrained=True).eval()
    # alexnet_dict = dict(
    #     type='alexnet',
    #     arch=alexnet,
    #     layer_name='features_10',
    #     input_size=(224,224)
    # )
    # test_cluster_scorecam(alexnet, alexnet_dict, img_file, "alexnet_scorecam", num_clusters=5)

    # # VGG16
    # vgg = models.vgg16(pretrained=True).eval()
    # vgg_dict = dict(
    #     type='vgg16',
    #     arch=vgg,
    #     layer_name='features_29',
    #     input_size=(224,224)
    # )
    # test_cluster_scorecam(vgg, vgg_dict, img_file, "vgg_scorecam", num_clusters=5)

    # ResNet18
    resnet = models.resnet18(pretrained=True).eval()
    resnet_dict = dict(
        type='resnet18',
        arch=resnet,
        layer_name='layer4',
        input_size=(224,224)
    )
    test_cluster_scorecam(resnet, resnet_dict, img_file, "resnet_scorecam", num_clusters=5)
