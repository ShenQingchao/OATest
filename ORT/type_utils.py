import numpy as np


onnx_type2numpy = {
    0: np.float32,  # UNDEFINED
    1: np.float32,  # FLOAT
    2: np.uint8,    # UINT8
    3: np.int8,     # INT8
    4: np.uint16,   # UINT16
    5: np.int16,    # INT16
    6: np.int32,    # INT32
    7: np.int64,    # INT64
    8: np.str_,     # STRING
    9: np.bool_,    # BOOL
    10: np.float16,  # FLOAT16
    11: np.float64,  # DOUBLE
    12: np.uint32,   # UINT32
    13: np.uint64,   # UINT64
    14: np.complex64,   # COMPLEX64
    15: np.complex128,  # COMPLEX128
    16: np.float16,  # BFLOAT16
    17: np.float32,  # FLOAT8E4M3FN (this one is not natively supported in NumPy, you might need custom handling)
    18: np.float32,  # FLOAT8E4M3FNUZ (same as above)
    19: np.float32,  # FLOAT8E5M2 (same as above)
    20: np.float32,  # FLOAT8E5M2FNUZ (same as above)
    21: np.uint8,    # UINT4 (this is tricky in NumPy, it's usually handled as a uint8)
    22: np.int8,     # INT4 (same as above)
}


if __name__ == '__main__':
    pass
