# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 128, 28, 28), dtype="float32"), w: R.Tensor((128, 16, 3, 3), dtype="float32")) -> R.Tensor((2, 128, 56, 84), dtype="float32"):
        gv: R.Tensor((2, 128, 56, 84), dtype="float32") = R.nn.conv2d_transpose(x, w, strides=[2, 3], padding=[1, 1, 1, 1], output_padding=[1, 2], dilation=[1, 1], groups=8, data_layout="NCHW", kernel_layout="IOHW", out_layout="NCHW", out_dtype="void")
        return gv