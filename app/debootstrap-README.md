# How to prepare your Debootstrap environment and use it to build

Install Deboostrap and QEMU User. Versions from Debian *experimental* repo (in the time of writing) are needed (otherwise QEMU-emulated programs can segfault and debootstrap will fail too). QEMU User enables you to run exeutables built for architectures not supported by your host machine *seamlessly*

## Bootstrap ARM64 Debian Buster basic system on your desktop

Issue following commands (you can replace `arm64-debootstrap` with another directory name).

```
sudo debootstrap --foreign --arch arm64 buster arm64-debootstrap/ http://deb.debian.org/debian
sudo chroot arm64-debootstrap/ /debootstrap/debootstrap --second-stage
```

Congratulations, you have basic Debian system on your desktop and if you succeeded in running the second `chroot` command, you also have just run ARM64 Bash seamlessly in order to install .deb ARM64 packages.

## Chrooting into ARM64 Debian

Issue following command

```
sudo chroot arm64-debootstrap/
```

Just to make sure everything is going to be in line with ARM64 Raspberry Pi OS prepend this line to

```
deb http://archive.raspberrypi.org/debian/ buster main
```

`/etc/apt/sources.list` file in chroot environment.

Fix keys https://chrisjean.com/fix-apt-get-update-the-following-signatures-couldnt-be-verified-because-the-public-key-is-not-available/ and run `sudo apt update`.

## Torchlib build prerequisites

Create fake `/proc/cpuinfo` file. Copy its contents from https://www.raspberrypi.org/forums/viewtopic.php?t=243882

```
apt install -y git cmake build-essential python3 libnuma-dev libpython3.7 python3-pip ninja-build ack python3-numpy
python3 -m pip install pyyaml typing_extensions
# apt install libideep-dev mkl
```

## Static Torchlib build

TLDR (more or less): follow https://github.com/pytorch/pytorch/blob/master/docs/libtorch.rst

```
export MAKEFLAGS=-j$(nproc)
git clone -b v1.8.0 --recurse-submodule https://github.com/pytorch/pytorch.git
mkdir pytorch-build-static
cd pytorch-build-static
cmake -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCXXFLAGS="-fPIC" -DCFLAGS="-fPIC" -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX:PATH=../pytorch-install-static ../pytorch
cmake --build . --target install
```

## Shared Torchlib build

```
export MAKEFLAGS=-j$(nproc)
git clone -b v1.8.0 --recurse-submodule https://github.com/pytorch/pytorch.git
mkdir pytorch-build-shared
cd pytorch-build-shared
cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../pytorch-install-shared ../pytorch
cmake --build . --target install
```

