FROM mcr.microsoft.com/devcontainers/anaconda:1-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
WORKDIR /tmp

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    dbus \
    fontconfig \
    gnupg \
    libasound2 \
    libfreetype6 \
    libglib2.0-0 \
    libnss3 \
    libsqlite3-0 \
    libx11-xcb1 \
    libxcb-glx0 \
    libxcb-xkb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxml2 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    libgl1-mesa-glx \
    libxkbfile-dev \
    openssh-client \
    wget \
    xcb \
    xkb-data && \
    apt-get clean

# QT6 is required for the Nsight Compute UI.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    qt6-base-dev && \
    apt-get clean

# Add NVIDIA's CUDA apt repository
# okay oppps this is very large
# RUN wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-debian12-12-8-local_12.8.1-570.124.06-1_amd64.deb
# RUN dpkg -i cuda-repo-debian12-12-8-local_12.8.1-570.124.06-1_amd64.deb
# RUN cp /var/cuda-repo-debian12-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/

# # Update the package lists again *after* adding the repository
# RUN apt-get update

# RUN apt-get -y install nsight-systems-cli


# Install Nsight Compute
# can update this with updated links/versions
RUN cd /tmp && \
    wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2023_4_1_97/nsight-systems-2023.4.1_2023.4.1.97-1_amd64.deb && \
    apt-get install -y ./nsight-systems-2023.4.1_2023.4.1.97-1_amd64.deb && \
    rm -rf /tmp/*
