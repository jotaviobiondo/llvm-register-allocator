# install deps
apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3 \
    subversion

# Getting sources & build & clean up
mkdir /usr/local/src/llvm-build \
  && cd /usr/local/src/llvm-build \
  && svn co http://llvm.org/svn/llvm-project/llvm/trunk llvm \
  && cd /usr/local/src/llvm-build/llvm/tools \
  && svn co http://llvm.org/svn/llvm-project/cfe/trunk clang \
  && cd /usr/local/src/llvm-build \
  && mkdir build \
  && cd build \
  && cmake -G "Unix Makefiles" ../llvm \
  && make \
  && make install \
  && cd ~/ \
  && rm -rf /usr/local/src/llvm-build
