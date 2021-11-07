# CryptoGRU

Gazelle MPC Framework with GRU

## Install

This was last tested on Ubuntu 16.04 LTS and 18.04 LST

```bash
  # Compile miracl for OSU cryptotools and boost
  cd third_party/cryptoTools/thirdparty/linux
  bash all.get

  # Compile cryptotools
  cd ../../
  cmake .
  make -j8

  # Compile gazelle
  cd ../../
  make -j8
```

If you want to run to run the network conversion scripts you will
need a python interpreter and pytorch. These scripts were tested with
Anaconda3 on a machine that had a GPU.

## Running examples

Have a look at the demo folder to see some examples.
