# coding=utf-8
import numpy as np


def patch_numpy_compat() -> None:
    replacements = {
        "bool_": bool,
        "int_": np.int64,
        "float_": np.float64,
        "complex_": np.complex128,
        "object_": object,
        "str_": str,
        "unicode_": str,
        "long": np.int64,
        "unicode": str,
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
    }
    for name, value in replacements.items():
        if not hasattr(np, name):
            setattr(np, name, value)
