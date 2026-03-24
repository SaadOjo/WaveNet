FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel


ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1


RUN apt update && apt install -y \
    python3-tk \
    x11-apps \
    xauth \
    x11-utils \
    tzdata \
    && apt clean

# RUN pip install --no-cache-dir typing_extensions==4.7.1 jupyterlab==4.3.4

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt


WORKDIR /workspace


CMD ["bash"]

