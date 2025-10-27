# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def fused_relax_nn_conv2d(data1: R.Tensor((1, 64, 56, 56), dtype="float32"), weight1: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
        R.func_attr({"Composite": "tensorrt.conv2d", "Primitive": 1})
        with R.dataflow():
            gv: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(data1, weight1, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            R.output(gv)
        return gv

    @R.function
    def main(data: R.Tensor((1, 64, 56, 56), dtype="float32"), weight: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((1, 64, 56, 56), dtype="float32") = cls.fused_relax_nn_conv2d(data, weight)
            conv: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv)
            R.output(conv)
        return conv