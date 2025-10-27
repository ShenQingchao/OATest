# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def fused_relax_nn_conv2d_relax_nn_relu(data1: R.Tensor((1, 64, 56, 56), dtype="float32"), weight11: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
        R.func_attr({"Composite": "dnnl.conv2d_relu", "Primitive": 1})
        with R.dataflow():
            lv1: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(data1, weight11, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            gv1: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv1)
            R.output(gv1)
        return gv1

    @R.function(private=True)
    def fused_relax_nn_conv2d_relax_nn_relu1(conv1: R.Tensor((1, 64, 56, 56), dtype="float32"), weight21: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Composite": "dnnl.conv2d_relu", "Primitive": 1})
        with R.dataflow():
            lv2: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(conv1, weight21, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            gv2: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(lv2)
            R.output(gv2)
        return gv2

    @R.function
    def main(data: R.Tensor((1, 64, 56, 56), dtype="float32"), weight1: R.Tensor((64, 64, 3, 3), dtype="float32"), weight2: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((1, 64, 56, 56), dtype="float32") = cls.fused_relax_nn_conv2d_relax_nn_relu(data, weight1)
            gv: R.Tensor((1, 64, 54, 54), dtype="float32") = cls.fused_relax_nn_conv2d_relax_nn_relu1(lv, weight2)
            R.output(gv)
        return gv