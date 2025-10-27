# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("n", "c", "h", "w"), dtype="float32"), kernel: R.Tensor(("f", "c", "kh", "kw"), dtype="float32")) -> R.Tensor(("n", "c", "h * 3 + kh - 3", "w * 3 + kw - 3"), dtype="float32"):
        n = T.int64()
        c = T.int64()
        h = T.int64()
        kh = T.int64()
        w = T.int64()
        kw = T.int64()
        f = T.int64()
        gv: R.Tensor((n, c, h * 3 + kh - 3, w * 3 + kw - 3), dtype="float32") = R.nn.conv2d_transpose(x, kernel, strides=[3, 3], padding=[0, 0, 0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="IOHW", out_layout="NCHW", out_dtype="void")
        return gv