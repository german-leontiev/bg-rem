"""
Main file. To start service run `uvicorn app:app --host 0.0.0.0 --port 5003`
"""
import flask
import torch
from bg_rem import remove_bg
from flask import render_template, request, send_file

app = flask.Flask(__name__)

# Preloading model for whole app to prevent loading for each user
model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")
model = model.eval().cpu()


@app.route("/", methods=["GET", "POST"])
def predict():
    """
    Simplest logic.
    If user loads page - sends template.
    If user sends image - removes bg from it
    """
    if request.method == "POST":
        f = request.files["file"]
        return send_file(remove_bg(f, model), as_attachment=True)

    return render_template("index.html")


app.run(host="0.0.0.0", port=5003)
