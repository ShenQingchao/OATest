# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def fused_relax_add(x11: R.Tensor((10,), dtype="float32"), x21: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"Composite": "compiler_A.add", "Primitive": 1})
        with R.dataflow():
            gv2: R.Tensor((10,), dtype="float32") = R.add(x11, x21)
            R.output(gv2)
        return gv2

    @R.function(private=True)
    def fused_relax_nn_gelu(x31: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"Composite": "compiler_B.gelu", "Primitive": 1})
        with R.dataflow():
            gv3: R.Tensor((10,), dtype="float32") = R.nn.gelu(x31)
            R.output(gv3)
        return gv3

    @R.function(private=True)
    def fused_relax_nn_relu(add2: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"Composite": "compiler_A.relu", "Primitive": 1})
        with R.dataflow():
            gv: R.Tensor((10,), dtype="float32") = R.nn.relu(add2)
            R.output(gv)
        return gv

    @R.function
    def main(x1: R.Tensor((10,), dtype="float32"), x2: R.Tensor((10,), dtype="float32"), x3: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((10,), dtype="float32") = cls.fused_relax_add(x1, x2)
            lv1: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_gelu(x3)
            lv11: R.Tensor((10,), dtype="float32") = cls.fused_relax_add(lv, lv1)
            lv12: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_gelu(lv11)
            lv2: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_relu(lv11)
            lv21: R.Tensor((10,), dtype="float32") = cls.fused_relax_add(lv12, lv2)
            gv1: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_relu(lv21)
            R.output(gv1)
        return gv1