# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def foo(x: R.Tensor((160,), dtype="float32")) -> R.Tensor((160,), dtype="float32"):
        with R.dataflow():
            lv1: R.Tensor((160,), dtype="float32") = R.arange(R.prim_value(0), R.prim_value(160), R.prim_value(1), dtype="float32")
            lv2: R.Tensor((160,), dtype="float32") = R.arange(R.prim_value(0), R.prim_value(160), R.prim_value(1), dtype="float32")
            lv3: R.Tensor((160,), dtype="float32") = R.add(x, lv1)
            out: R.Tensor((160,), dtype="float32") = R.add(lv3, lv2)
            R.output(out)
        return out