# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("n", "c", "h", "w"), dtype="float32"), kernel: R.Tensor(("f", "c", "kh", "kw"), dtype="float32")) -> R.Tensor(("n", "f", "h - kh + 1", "w - kw + 1"), dtype="float32"):
        n = T.int64()
        f = T.int64()
        h = T.int64()
        kh = T.int64()
        w = T.int64()
        kw = T.int64()
        c = T.int64()
        gv: R.Tensor((n, f, h - kh + 1, w - kw + 1), dtype="float32") = R.nn.conv2d(x, kernel, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
        return gv