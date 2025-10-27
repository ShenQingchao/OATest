# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 28, 128), dtype="float32"), w: R.Tensor((64, 128, 3), dtype="float32")) -> R.Tensor((2, 26, 64), dtype="float32"):
        gv: R.Tensor((2, 26, 64), dtype="float32") = R.nn.conv1d(x, w, strides=[1], padding=[0, 0], dilation=[1], groups=1, data_layout="NWC", kernel_layout="OIW", out_layout="NWC", out_dtype="void")
        return gv