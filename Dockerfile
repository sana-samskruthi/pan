FROM python:3.10

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

RUN apt-get install -y tesseract-ocr python3 python3-pip

# ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 4000

CMD ["python3", "pan.py"]