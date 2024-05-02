# RHEL9.3 UBI
FROM registry.access.redhat.com/ubi9/ubi:9.3-1610

USER root

EXPOSE 5000

# Update image
RUN yum update --disableplugin=subscription-manager -y && rm -rf /var/cache/yum && yum clean all -y

# Install libraries using yum
RUN yum install -y git wget gcc gcc-c++ libgcc cmake make libjpeg-turbo --disableplugin=subscription-manager && rm -rf /var/cache/yum && yum clean all -y

# Install Miniconda3
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-ppc64le.sh
RUN chmod a+x Miniconda3-latest-Linux-ppc64le.sh && \
    mkdir -p /root/.conda
RUN bash ./Miniconda3-latest-Linux-ppc64le.sh -b

ENV PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN conda update -n base -c defaults conda --yes

RUN conda init bash && \
    cat /root/.bashrc && \
    conda config --add channels conda-forge

RUN source /root/.bashrc && \
    conda create -n icp python=3.7 --yes

RUN source /root/.bashrc && \
    conda activate icp

ENV PATH=/root/miniconda3/envs/icp/bin:/root/miniconda3/condabin:/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN source /root/.bashrc && \
    conda activate icp && \
    conda install -c conda-forge scikit-learn=0.21.2 --yes && \
    conda install -c conda-forge numpy --yes && \
    conda install -c conda-forge flask --yes

# 2 python files used by the REST API
RUN mkdir /model
ADD model/* /model/
ADD pred_test.py /

CMD [ "python", "/pred_test.py" ]
