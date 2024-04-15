---
title: "3D Experience"
date: 2024-04-15T15:06:23+08:00
draft: true
---
# Motivation
本文章用来总结本人在3D视觉研究中遇到的问题，同时给出一些经验，主要与Python，NeRF，3DGS相关。
# 安装常见问题
## 基本网络问题
一定要梯子！一定要梯子！一定要梯子！

梯子可以解决99%的pip install和conda install遇到的问题
当然，考虑到一些公司环境，这里还是提供一些github镜像和pip镜像（清华源）以及conda镜像（清华源）

清华源tuna是一个非常好的项目，偶尔会封禁一些下载流量过大的ip，这个时候可以发邮件向他们申请，回复及时且很友善。


清华pip源配置命令
```
参考https://mirrors.tuna.tsinghua.edu.cn/help/pypi/
```
临时使用：
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 你的包
```
默认使用（pip版本>=10.0.0）
```
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

清华conda源配置命令
```
参考https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/
```
修改你的`/home/用户名/.condarc`文件，替换为
```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  deepmodeling: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/
```
如果没有这个文件，则直接创建，无法创建的windows用户可使用`conda config --set show_channel_urls yes`

## Torch版本老是对不上或者没有CUDA

很多人都被这个问题折磨过，不过这里给出一个比较通用的解决方案
- 首先，明确你需要的torch版本，torchvision版本和你自己的cuda版本`nvcc -V`
- 接下来，有多种选择
  - 使用conda安装：
    ```
    conda install pytorch=XXX torchvision=XXX cudatoolkit=XXX -c pytorch`
    ```
  - 使用conda的好处是简单、高效，且可以自由选择cuda的版本，不过大多数人的网络可能不支持这么丝滑的下载
  - 使用pytorch官方网站的下载连接，手动下载正确版本：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

## 个别较难安装的库，如pytorch3D
这里仅以pytorch3D举例