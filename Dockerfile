#TODO: specify small size
FROM python

WORKDIR /usr/src
COPY requirements.txt .
RUN pip --no-cache install -r requirements.txt

COPY app.py .
COPY cache_model.py .
COPY templates templates
COPY static static

CMD ["python3", "./cache_model.py"]
CMD ["python3", "./app.py"]
