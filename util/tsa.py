"""
Name: tsa
Date: 2020/2/29 下午2:52
"""

import numpy as np


def get_tsa(tsa_type, train_step, total_step, start=0, end=1):
    tsa_result = 0
    if tsa_type == 'log':
        tsa_result = (1 - np.exp(-(train_step / total_step) * 5)) * (1 - 1/total_step) + 1/total_step
    elif tsa_type == 'exp':
        tsa_result = np.exp(((train_step/total_step) - 1) * 5) * (1 - 1/total_step) + 1/total_step
    elif tsa_type == 'linear':
        tsa_result = (train_step/total_step) * (1 - 1/total_step) + 1/total_step

    return tsa_result * (end - start) + start
