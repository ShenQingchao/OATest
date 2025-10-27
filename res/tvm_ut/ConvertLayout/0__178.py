# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")) -> R.Tensor((2, 2, 9, 7), dtype="float32"):
        with R.dataflow():
            gv: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(x, w, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            gv2: R.Tensor((2, 2, 9, 7), dtype="float32") = R.strided_slice(gv, (R.prim_value(1), R.prim_value(2), R.prim_value(3)), (R.prim_value(0), R.prim_value(0), R.prim_value(0)), (R.prim_value(4), R.prim_value(26), R.prim_value(26)), (R.prim_value(2), R.prim_value(3), R.prim_value(4)), assume_inbound=False)
            R.output(gv2)
        return gv2