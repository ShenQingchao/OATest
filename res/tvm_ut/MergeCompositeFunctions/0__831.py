# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def fused_relax_add(lv: R.Tensor((10,), dtype="float32"), gelu1: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"Composite": "compiler_A.add", "Primitive": 1})
        with R.dataflow():
            gv: R.Tensor((10,), dtype="float32") = R.add(lv, gelu1)
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_nn_gelu(x21: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"Composite": "compiler_A.gelu", "Primitive": 1})
        with R.dataflow():
            gv3: R.Tensor((10,), dtype="float32") = R.nn.gelu(x21)
            R.output(gv3)
        return gv3

    @R.function(private=True)
    def fused_relax_nn_relu(x11: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"Composite": "compiler_A.relu", "Primitive": 1})
        with R.dataflow():
            gv2: R.Tensor((10,), dtype="float32") = R.nn.relu(x11)
            R.output(gv2)
        return gv2

    @R.function
    def main(x1: R.Tensor((10,), dtype="float32"), x2: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv1: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_relu(x1)
            lv2: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_gelu(x2)
            lv3: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_relu(lv1)
            lv4: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_gelu(lv2)
            gv1: R.Tensor((10,), dtype="float32") = cls.fused_relax_add(lv3, lv4)
            R.output(gv1)
        return gv1