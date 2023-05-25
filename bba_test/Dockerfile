# 基础镜像， cuda为11.0，ubuntu18.04
FROM image.rsaicp.com/base/cuda:11.0-cudnn8-runtime-ubuntu18.04

# 配置程序依赖环境
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         curl \
         ca-certificates \
         libgl1-mesa-glx \
         libjpeg-dev \
         libpng-dev \
         libgtk2.0-dev \
         python3.8-venv \
         python3.8-dev \
         python3-pip && \
     rm -rf /var/lib/apt/lists/*
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN python3 --version
RUN export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
#安装环境依赖
RUN pip3 install --upgrade pip
RUN pip install setuptools
RUN pip install wheel
COPY requirements.txt /tmp/
RUN pip3 install --requirement /tmp/requirements.txt

#RUN pip install torch==1.8.0+cu110 torchvision==0.9.0+cu110 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch==1.7.1+cu110 torchvision===0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip3 install torch==1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

#RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip3 install torch==1.7.0+cu110 torchvision===0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
# 将程序复制到容器内的/work路径下
RUN mkdir /work
WORKDIR /work
COPY .  /work

RUN mkdir /input_path
RUN chmod 444 /input_path

RUN mkdir /output_path
RUN chmod 666 /output_path

# 容器启动命令
CMD ["python3", "-u","/work/main.py"]