FROM python:3.7

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt

WORKDIR /app
COPY *.py /app/
ENTRYPOINT ["python", "sift.py"]
