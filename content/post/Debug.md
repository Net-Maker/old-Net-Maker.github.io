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

新增：这个报错还可能发生在你的光栅化没有正确与你的torch版本和cuda版本对应上的时候，此时：
- 使用nvcc -V检查你的cuda版本
- 使用pip list | grep torch检查你的torch版本是否对应好了cuda版本
- 删除submodule/diff-gaussian-rasterization/build，然后在/submodule/diff-gaussian-rasterization这个目录下运行 python setup.py install。

以上步骤基本可以解决问题，注意此方法适用于在跑gaussian相关baseline的时候，初次配环境出的错，如果是你后期改代码导致报错了，还是关注你自己写的代码有没有问题吧，或者检查你系统的CUDA版本是不是改变了。