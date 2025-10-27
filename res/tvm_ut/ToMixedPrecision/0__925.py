# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((3, 3, 3, 3), dtype="float32")) -> R.Tensor((2, 3, 26, 26), dtype="float32"):
        with R.dataflow():
            gv: R.Tensor((2, 3, 26, 26), dtype="float32") = R.nn.conv2d(x, w, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            gv1: R.Tensor((2, 3, 26, 26), dtype="float32") = R.nn.softmax(x, axis=1)
            gv2: R.Tensor((2, 3, 26, 26), dtype="float32") = R.add(gv, gv1)
            R.output(gv2)
        return gv2