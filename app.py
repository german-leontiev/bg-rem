
from pathlib import Path
import flask
from flask import render_template, request, abort, redirect, send_file
from werkzeug.utils import secure_filename

import torch
from PIL import Image
import torchvision.transforms.functional as TF

app = flask.Flask(__name__)

model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")
model = model.eval().cpu()


def rem_bg(pth_, model):
    bgr = torch.tensor([0.47, 1, 0.6]).view(3, 1, 1).cpu()  # Green background.
    rec = [None] * 4  # Initial recurrent states.
    down_sample_ratio = 0.25
    image = Image.open(pth_)
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    assert len(x.shape) == 4, "Input tensor must have 4 channels"
    fgr, pha, *rec = model(x.cpu(), *rec, down_sample_ratio)
    assert len(pha.shape) == 4, "Prediction error. Wrong # of channels"
    np_arr_alpha = pha.cpu().detach().numpy()[0, 0, :, :]
    alpha = Image.fromarray((np_arr_alpha * 255).astype("uint8"))
    image.putalpha(alpha)
    return image


def rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()
    assert not pth.exists(), "Filesystem cleaning error"


@app.route("/", methods=["GET", "POST"])
def predict():
    # When user sends file
    if request.method == "POST":
        # Check request
        assert "file" in request.files, "File has not provided in request"
        f = request.files["file"]

        # Delete and recreate file  after last use
        folder_name = "img/uploaded/"
        Path("img").mkdir(parents=True, exist_ok=True)
        rm_tree("img")
        Path(folder_name).mkdir(parents=True, exist_ok=True)
        assert Path(folder_name).exists(), "Directory creation error"

        filepath = folder_name + secure_filename(f.filename)
        f.save(filepath)
        image = rem_bg(filepath, model)
        save_path = "static/result.png"
        image.save(save_path)
        assert Path(save_path).exists(), "Image was not saved in filesystem"

        return send_file(save_path, as_attachment=True)

    return render_template("index.html", predicted=True)


app.run(host="0.0.0.0", port=5003)
