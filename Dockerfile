from python:3.11-slim-buster
WORKDIR / application
COPY ./ application
RUN apt update -y && apt install awscli -y
# COPY requirements.txt /tmp/requirements.txt
# RUN python3 -m pip install -r /tmp/requirements.txt

CMD ["python3","application.py"]