# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
        with R.dataflow():
            lv0: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
            lv1: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
            gv: R.Tensor((2, 3), dtype="float32") = R.multiply(lv0, lv1)
            R.output(gv)
        return gv