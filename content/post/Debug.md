---
title: "Debug记录"
date: 2024-04-15T23:29:29+08:00
---
### 本文用来记录我遇到的各种奇葩bug，不定时更新
# Hugo
- git config --global core.autocrlf false, 如果使用Windows和git配置hugo要记得这个，不能修改为CRLF，小心这个warning：LF will be replaced by CRLF the next time Git touches it

# CUDA
- ## 运行3DGS相关代码时出现 CUDA error : illegal memory access
这个问题存在很多争议，这里只讨论3D高斯相关代码中这个问题的情况，我是自己魔改代码时出现的.

主要问题就是你有数据在cpu上，有数据在gpu上，有的作者喜欢在cpu上渲染（少数），有的喜欢丢到gpu上渲染。

这个报错最难受的就是无法定位到具体的错误，这个时候可以使用`torch.cuda.synchronize()`往前一行一行地执行来定位具体错误，也可以直接打印变量的device，然后指定好是用`.cpu()`还是`.cuda()`.