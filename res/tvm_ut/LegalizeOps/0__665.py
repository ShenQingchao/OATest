# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 128, 28, 28), dtype="float32"), w: R.Tensor((64, 16, 3, 3), dtype="float32")) -> R.Tensor((2, 64, 13, 13), dtype="float32"):
        gv: R.Tensor((2, 4, 13, 13), dtype="float32") = R.nn.conv2d(x, w, strides=[2, 2], padding=[1, 1, 1, 1], dilation=[2, 2], groups=8, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
        return gv