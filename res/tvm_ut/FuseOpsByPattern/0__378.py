# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 10), dtype="float32"):
        with R.dataflow():
            gv: R.Tensor((10, 10), dtype="float32") = R.clip(x, R.prim_value(0), R.prim_value(4))
            R.output(gv)
        return gv