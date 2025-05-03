#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# https://stackoverflow.com/questions/60057795/attributeerror-module-scipy-stats-has-no-attribute-signaltonoise
#
# 下面是一个使用SciPy计算信噪比的示例:
import numpy as np
import scipy.stats as stats

import numpy as np
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


# 生成随机信号数据和噪声数据
signal = np.random.randn(100)
noise = np.random.randn(100)

# 计算信噪比
#snr = stats.signaltonoise(signal, noise)
snr = signaltonoise(signal)

# 打印结果
print("信噪比:", snr)

