FROM python:3.8

WORKDIR /usr/src
COPY requirements.txt .
RUN pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip --no-cache install -r requirements.txt

COPY app.py .
COPY bg_rem.py bg_rem.py
COPY cache_model.py .
COPY templates templates
COPY static static

CMD ["python3", "./cache_model.py"]
CMD ["python3", "./app.py"]
