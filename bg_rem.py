"""
App logic.
Getting, processing and sending image file.
"""
from pathlib import Path
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms.functional as TF
import torch


def remove_bg(f, model):
    """
    Pipeline of file processing
    :param f: input file
    :param model: input preload model
    :return: path to processed image
    """
    folder_name = "img/uploaded/"
    Path("img").mkdir(parents=True, exist_ok=True)
    rm_tree("img")
    Path(folder_name).mkdir(parents=True, exist_ok=True)

    filepath = folder_name + secure_filename(f.filename)
    f.save(filepath)
    image = rem_bg(filepath, model)
    save_path = "static/result.png"
    image.save(save_path)
    return save_path


def rem_bg(pth_, model):
    """
    Uses NN to remove background from portrait
    :param pth_: path to saved file
    :param model: preloaded model
    :return: processed image
    """
    bgr = torch.tensor([0.47, 1, 0.6]).view(3, 1, 1).cpu()  # Green background.
    rec = [None] * 4  # Initial recurrent states.
    downsample_ratio = 0.25
    image = Image.open(pth_)
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    fgr, pha, *rec = model(x.cpu(), *rec, downsample_ratio)
    np_arr_alph = pha.cpu().detach().numpy()[0, 0, :, :]
    alpha = Image.fromarray((np_arr_alph * 255).astype("uint8"))
    image.putalpha(alpha)
    return image


def rm_tree(pth):
    """
    This function cleans dir for next user
    :param pth: path to dir
    :return: None
    """
    pth = Path(pth)
    for child in pth.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()
