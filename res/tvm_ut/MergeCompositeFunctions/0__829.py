# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def fused_relax_add(lv5: R.Tensor((1, 64, 54, 54), dtype="float32"), gelu1: R.Tensor((1, 64, 54, 54), dtype="float32")) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Composite": "compiler_A.add", "Primitive": 1})
        with R.dataflow():
            gv3: R.Tensor((1, 64, 54, 54), dtype="float32") = R.add(lv5, gelu1)
            R.output(gv3)
        return gv3

    @R.function(private=True)
    def fused_relax_nn_conv2d(data1: R.Tensor((1, 64, 56, 56), dtype="float32"), weight1: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Composite": "compiler_A.conv2d", "Primitive": 1})
        with R.dataflow():
            gv4: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(data1, weight1, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            R.output(gv4)
        return gv4

    @R.function(private=True)
    def fused_relax_nn_gelu(lv: R.Tensor((1, 64, 54, 54), dtype="float32")) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Composite": "compiler_B.gelu", "Primitive": 1})
        with R.dataflow():
            gv: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.gelu(lv)
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_nn_relu(lv1: R.Tensor((1, 64, 54, 54), dtype="float32")) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Composite": "compiler_A.relu", "Primitive": 1})
        with R.dataflow():
            gv1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(lv1)
            R.output(gv1)
        return gv1

    @R.function
    def main(data: R.Tensor((1, 64, 56, 56), dtype="float32"), weight: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv2: R.Tensor((1, 64, 54, 54), dtype="float32") = cls.fused_relax_nn_conv2d(data, weight)
            lv3: R.Tensor((1, 64, 54, 54), dtype="float32") = cls.fused_relax_nn_relu(lv2)
            lv4: R.Tensor((1, 64, 54, 54), dtype="float32") = cls.fused_relax_nn_gelu(lv2)
            gv2: R.Tensor((1, 64, 54, 54), dtype="float32") = cls.fused_relax_add(lv3, lv4)
            R.output(gv2)
        return gv2