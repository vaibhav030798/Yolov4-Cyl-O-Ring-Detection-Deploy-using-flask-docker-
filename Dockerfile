FROM continuumio/anaconda3:latest

WORKDIR /home

RUN apt-get update && apt-get install -y python3-opencv

RUN pip install opencv-python

COPY Cylinder-O-Ring .

CMD ["python", "app.py"]

