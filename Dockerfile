FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update \
    && apt install -y language-pack-ja-base \
    && update-locale LANG=ja_JP.UTF-8 \
    && apt-get install -y python3 python3-pip \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 0 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 0 \
    && pip install --upgrade pip

RUN apt-get install -y build-essential libffi-dev libssl-dev zlib1g-dev liblzma-dev libbz2-dev libreadline-dev libsqlite3-dev git curl wget
RUN git clone https://github.com/pyenv/pyenv.git /root/.pyenv \
    && echo 'export PYENV_ROOT="/root/.pyenv"' >> /root/.bashrc \
    && echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> /root/.bashrc \
    && echo 'eval "$(pyenv init -)"' >> /root/.bashrc

SHELL ["/bin/bash", "-c"]

ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"

RUN eval "$(pyenv init -)" \
    && pyenv install 3.6.12 \
    && pyenv global 3.6.12 \
    && pip install -U pip

COPY torch-1.1.0-cp36-cp36m-linux_x86_64.whl /tmp/

RUN eval "$(pyenv init -)" \
    && pip install /tmp/torch-1.1.0-cp36-cp36m-linux_x86_64.whl

RUN eval "$(pyenv init -)" \
    && pip install matplotlib pandas pillow torchvision==0.3.0 scipy

CMD ["/bin/bash"]

