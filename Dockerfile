# FROM nvidia/cuda:12.2.0-base-ubuntu20.04
FROM nvidia/cuda:12.6.3-base-ubuntu22.04

WORKDIR /workspace

COPY . /workspace

RUN set -xe \
    && ls -a /etc/apt/sources.list.d/ \
    && rm /etc/apt/sources.list.d/cuda* \
    && apt-get update -y \
    && apt-get install -y python3-pip \
    && python3 --version \
    && python3 -m pip --version \
    && python3 -m pip install --no-cache-dir -r requirements.txt

# Run a command when the container starts (optional)
# CMD ["bash"]