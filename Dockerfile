FROM ubuntu:18.04

LABEL author="Bo Feng" email="fengbo@iu.edu"

WORKDIR /cryptogru

COPY . /cryptogru/

# install building toolkits
RUN apt-get update && apt-get install tree build-essential cmake nasm libboost-all-dev -y

# Compile cryptotools
RUN cd /cryptogru/third_party/cryptoTools; \
	make clean; \
	rm -rf CMakeCache.txt CMakeFiles; \
	cd /cryptogru/third_party/cryptoTools/thirdparty/linux; \
	/bin/bash miracl.get; \
	cd /cryptogru/third_party/cryptoTools; \
	cmake .; \
	make -j4; \
	ls -l lib; \
	ls -l bin

# Compile gazelle
RUN cd /cryptogru/; \
	make -j4; \
	tree bin