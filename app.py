import subprocess
import sys
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
    bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cpu()  # Green background.
    rec = [None] * 4                                       # Initial recurrent states.
    downsample_ratio = 0.25
    image = Image.open(pth_)
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    fgr, pha, *rec = model(x.cpu(), *rec, downsample_ratio)
    np_arr_alph =  pha.cpu().detach().numpy()[0,0,:,:]
    alpha = Image.fromarray((np_arr_alph*255).astype('uint8'))
    image.putalpha(alpha)
    return image

def rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


@app.route("/", methods=["GET", "POST"])
def predict():
    # When user sends file
    if request.method == "POST":
        # Check request
        if "file" not in request.files:
            return redirect(request.url)
        f = request.files["file"]
        if not f:
            return

        # Delete and recreate file  after last use
        foldername = "img/uploaded/"
        Path("img").mkdir(parents=True, exist_ok=True)
        rm_tree("img")
        Path(foldername).mkdir(parents=True, exist_ok=True)
        
        filepath = foldername + secure_filename(f.filename)
        f.save(filepath)
        image = rem_bg(filepath, model)
        save_path = 'static/result.png'
        image.save(save_path)
#         return render_template("index.html", predicted=True)
        return send_file(save_path, as_attachment=True)

    return render_template("index.html")


app.run(host="0.0.0.0", port=5003)
