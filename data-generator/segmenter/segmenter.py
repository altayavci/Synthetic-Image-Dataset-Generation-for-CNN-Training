import os
from huggingface_hub import hf_hub_download

from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = None
ISNetDIS = None
normalize = None
im_preprocess = None
hypar = None
net = None


def init():
    global device, ISNetDIS, normalize, im_preprocess, hypar, net

    if not os.path.exists("segmenter/saved_models"):
        os.makedirs("./segmenter/saved_models")
        hf_hub_download(
            repo_id="NimaBoscarino/IS-Net_DIS-general-use",
            filename="isnet-general-use.pth",
            local_dir="segmenter/saved_models",
        )
        os.system("rm -r segmenter/git/xuebinqin/DIS/IS-Net/__pycache__")

    import segmenter.models as models
    import segmenter.data_loader_cache as data_loader_cache

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ISNetDIS = models.ISNetDIS
    normalize = data_loader_cache.normalize
    im_preprocess = data_loader_cache.im_preprocess

    hypar = {}
    hypar["model_path"] = "segmenter/saved_models"
    hypar["restore_model"] = "isnet-general-use.pth"
    hypar["interm_sup"] = False
    hypar["model_digit"] = "full"
    hypar["seed"] = 0
    hypar["cache_size"] = [1024, 1024]
    hypar["input_size"] = [1024, 1024]
    hypar["crop_size"] = [1024, 1024]

    hypar["model"] = ISNetDIS()
    net = build_model(hypar, device)


class GOSNormalize(object):
    """
    Normalize the Image using torch.transforms
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        return image


transform = transforms.Compose([GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])


def load_image(im_pil, hypar):
    im = np.array(im_pil)
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))

    return transform(im).unsqueeze(0), shape.unsqueeze(0)


def build_model(hypar, device):
    net = hypar["model"]
    if hypar["model_digit"] == "half":
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)

    if hypar["restore_model"] != "":
        net.load_state_dict(
            torch.load(
                hypar["model_path"] + "/" + hypar["restore_model"], map_location=device
            )
        )
        net.to(device)
    net.eval()
    return net


def predict(net, inputs_val, shapes_val, hypar, device):
    """
    Given an Image, predict the mask
    """
    net.eval()

    if hypar["model_digit"] == "full":
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)

    inputs_val_v = Variable(inputs_val, requires_grad=False).to(device)

    ds_val = net(inputs_val_v)[0]
    pred_val = ds_val[0][0, :, :, :]
    pred_val = torch.squeeze(
        F.upsample(
            torch.unsqueeze(pred_val, 0),
            (shapes_val[0][0], shapes_val[0][1]),
            mode="bilinear",
        )
    )

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val - mi) / (ma - mi)

    if device == "cuda":
        torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy() * 255).astype(np.uint8)


def segment(image):
    image_tensor, orig_size = load_image(image, hypar)
    mask = predict(net, image_tensor, orig_size, hypar, device)

    mask = Image.fromarray(mask).convert("L")
    im_rgb = image.convert("RGB")

    cropped = im_rgb.copy()
    cropped.putalpha(mask)

    return [cropped, mask]
