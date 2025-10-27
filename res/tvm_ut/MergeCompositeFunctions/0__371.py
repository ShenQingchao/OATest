# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def fused_relax_multiply(x: R.Tensor((10, 10), dtype="float32"), y: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 10), dtype="float32"):
        R.func_attr({"Composite": "cutlass.multiply", "Primitive": 1})
        with R.dataflow():
            gv: R.Tensor((10, 10), dtype="float32") = R.multiply(x, y)
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((10, 10), dtype="float32"), y: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 10), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv: R.Tensor((10, 10), dtype="float32") = cls.fused_relax_multiply(x, y)
            R.output(gv)
        return gv