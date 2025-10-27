# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 128, 28), dtype="float32"), w: R.Tensor((64, 16, 3), dtype="float32")) -> R.Tensor((2, 64, 13), dtype="float32"):
        gv: R.Tensor((2, 4, 13), dtype="float32") = R.nn.conv1d(x, w, strides=[2], padding=[1, 1], dilation=[2], groups=8, data_layout="NCW", kernel_layout="OIW", out_layout="NCW", out_dtype="void")
        return gv