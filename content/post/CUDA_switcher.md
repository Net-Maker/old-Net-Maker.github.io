---
title: "CUDA_switcher"
date: 2024-07-09T13:26:31+08:00
draft: false
---

最近写了一个bash脚本，可以方便地切换CUDA版本的同时不会把你的PATH变得很大，在多个不同cuda版本编译的conda环境中切换非常方便，分享一下：

创建cuda_switcher.sh:
```
# ~/cuda-switcher/switch_cuda.sh
#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: source switch_cuda.sh <version>"
    echo "Example: source switch_cuda.sh 11.2"
    return 1
fi

CUDA_VERSION=$1
CUDA_PATH="/usr/local/cuda-$CUDA_VERSION"

if [ ! -d "$CUDA_PATH" ]; then
    echo "CUDA version $CUDA_VERSION not found at $CUDA_PATH"
    return 1
fi

# Remove any existing CUDA paths from PATH and LD_LIBRARY_PATH
export PATH=$(echo $PATH | sed -e 's|/usr/local/cuda-[^/]*||g' -e 's|::|:|g' -e 's|^:||' -e 's|:$||')
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed -e 's|/usr/local/cuda-[^/]*||g' -e 's|::|:|g' -e 's|^:||' -e 's|:$||')

# Add the new CUDA paths
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "Switched to CUDA $CUDA_VERSION"

```

使用： 
- 首先确定你的CUDA_VERSION，此脚本不会检测你电脑上是否存在这一版本，不过输错了也不怕，再输一次对的就好了。
- 然后运行`bash cuda_switcher.sh $YOUR_CUDA_VERSION`