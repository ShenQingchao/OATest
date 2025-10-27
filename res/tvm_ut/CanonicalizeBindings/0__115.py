# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor) -> R.Tensor:
        with R.dataflow():
            lv1: R.Tensor = R.add(x, R.const(1, "int32"))
            gv1: R.Tensor = lv1
            gv2: R.Tensor = R.add(lv1, R.const(1, "int32"))
            R.output(gv1, gv2)
        return gv2