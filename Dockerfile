FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y apt-transport-https
RUN echo 'deb http://private-repo-1.hortonworks.com/HDP/ubuntu14/2.x/updates/2.4.2.0 HDP main' >> /etc/apt/sources.list.d/HDP.list
RUN echo 'deb http://private-repo-1.hortonworks.com/HDP-UTILS-1.1.0.20/repos/ubuntu14 HDP-UTILS main'  >> /etc/apt/sources.list.d/HDP.list
RUN echo 'deb [arch=amd64] https://apt-mo.trafficmanager.net/repos/azurecore/ trusty main' >> /etc/apt/sources.list.d/azure-public-trusty.list

# Install python2
RUN apt-get install -y python2.7 python-dev python-setuptools git cmake wget subversion libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN easy_install pip==9.0.3

# Install MKL blas
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
RUN apt-get update
RUN apt-get install -y intel-mkl-64bit-2018.2-046
RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so     libblas.so-x86_64-linux-gnu      /opt/intel/mkl/lib/intel64/libmkl_rt.so 150
RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so.3   libblas.so.3-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 150
RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so   liblapack.so-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 150
RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so.3 liblapack.so.3-x86_64-linux-gnu /opt/intel/mkl/lib/intel64/libmkl_rt.so 150
RUN echo "/opt/intel/lib/intel64"     >  /etc/ld.so.conf.d/mkl.conf
RUN echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/mkl.conf
ENV CPATH /opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/include/:$CPATH
RUN ldconfig

WORKDIR /tmp
# Install the right protobuf
RUN apt-get install -y autoconf automake libtool curl make g++ unzip zlib1g-dev
RUN curl -OL https://github.com/google/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
RUN unzip protoc-3.6.1-linux-x86_64.zip -d protoc3
RUN mv protoc3/bin/* /usr/local/bin/ 
RUN mv protoc3/include/* /usr/local/include/

RUN curl -OL http://ftp.se.debian.org/debian/pool/main/p/protobuf/libprotobuf-lite17_3.6.1.3-2_amd64.deb
RUN dpkg -i libprotobuf-lite17_3.6.1.3-2_amd64.deb

RUN curl -OL http://ftp.se.debian.org/debian/pool/main/p/protobuf/libprotobuf17_3.6.1.3-2_amd64.deb
RUN dpkg -i libprotobuf17_3.6.1.3-2_amd64.deb

RUN curl -OL http://ftp.se.debian.org/debian/pool/main/p/protobuf/libprotobuf-dev_3.6.1.3-2_amd64.deb
RUN dpkg -i libprotobuf-dev_3.6.1.3-2_amd64.deb

RUN echo "MKL_THREADING_LAYER=GNU" >> /etc/environment

# Use Caffe2 image as parent image
RUN pip install torch torchvision
ENV PATH /usr/local/lib/python2.7/dist-packages/torch:${PATH}
ENV Caffe2_DIR /usr/local/lib/python2.7/dist-packages/torch/share/cmake/Caffe2/
# These files are missing for some reason
RUN cd /usr/local/lib/python2.7/dist-packages/torch/include/caffe2/utils/
RUN svn checkout https://github.com/pytorch/pytorch/trunk/caffe2/utils/
RUN mv -vn ./utils/* /usr/local/lib/python2.7/dist-packages/torch/include/caffe2/utils
RUN ls /usr/local/lib/python2.7/dist-packages/torch/include/caffe2/utils/math

# Clone the Detectron repository
RUN git clone https://github.com/facebookresearch/densepose /densepose
RUN cd /densepose && git checkout 35e69d1
ENV DENSEPOSE /densepose

# Install Python dependencies
RUN pip install -r /densepose/requirements.txt

# A line is wrong in densepose, because of course it is.
RUN sed -i '54 c\ \ \ \ prefixes = [_CMAKE_INSTALL_PREFIX, sys.prefix, sys.exec_prefix, "/usr/local/lib/python2.7/dist-packages/torch"] + sys.path' /densepose/detectron/utils/env.py


# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
RUN cd /cocoapi && git checkout aca78bc
WORKDIR /cocoapi/PythonAPI
RUN make install 

# Go to Densepose root
wORKDIR /densepose

# Set up Python modules
RUN make

# [Optional] Build custom ops
RUN make ops

# Added by Axel:

# Get other needed densepose data
RUN cd /densepose/DensePoseData && sh get_densepose_uv.sh

# Install other packages
RUN apt-get install -y ffmpeg nano
RUN pip install numpy SimpleWebSocketServer opencv-python pyyaml==3.12 posix_ipc lmdb scikit-learn scikit-cuda colormath youtube-dl pafy ffmpeg-python
RUN pip install -U protobuf

RUN rm -rf /tmp && cd /
