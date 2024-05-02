# RHEL9.3 UBI
FROM registry.access.redhat.com/ubi9/ubi:9.3-1610

# Update image
RUN yum update --disableplugin=subscription-manager -y && rm -rf /var/cache/yum && yum clean all -y

# Install libraries using yum
RUN yum install -y git wget gcc gcc-c++ libgcc cmake make libjpeg-turbo --disableplugin=subscription-manager && rm -rf /var/cache/yum && yum clean all -y

# Get Anaconda3 scrtip for ppc64le
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-ppc64le.sh

RUN chmod a+x Anaconda3-2023.09-0-Linux-ppc64le.sh && \
   mkdir -p /root/.conda

RUN bash ./Anaconda3-2023.09-0-Linux-ppc64le.sh -b

ENV PATH=/root/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN conda update -n base -c defaults conda --yes

RUN conda init bash && \
    cat /root/.bashrc && \
    conda config --add channels conda-forge && \
    conda config --add channels rocketce

RUN source /root/.bashrc && \
    conda create -n lstm python=3.9 --yes && \
    conda activate lstm

ENV PATH=/root/anaconda3/envs/lstm/bin:/root/anaconda3/condabin:/root/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN source /root/.bashrc && \
    conda activate lstm && \
    conda install scikit-learn=1.0.2 -y && \
    conda install pydot -y && \
    conda install tensorflow-cpu -y && \
    conda install pandas -y && \
    conda install sklearn-pandas && \
    conda install flask

RUN source /root/.bashrc && \
    conda env list && \
    conda activate lstm && \
    conda list

EXPOSE 5000

RUN mkdir /extra
ADD extra/* /extra/
ADD flask_inf.py /

CMD ["python", "flask_inf.py"]
