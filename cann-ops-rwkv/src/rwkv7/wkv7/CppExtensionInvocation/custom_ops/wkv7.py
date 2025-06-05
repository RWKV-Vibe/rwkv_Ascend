import torch
import custom_ops_lib


def wkv7(k, v, w, r, a, b, hi):
    return custom_ops_lib.wkv7(k, v, w, r, a, b, hi)
