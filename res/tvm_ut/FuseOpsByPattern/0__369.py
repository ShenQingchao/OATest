# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((10, 10), dtype="float32"), y: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 10), dtype="float32"):
        with R.dataflow():
            gv: R.Tensor((10, 10), dtype="float32") = R.multiply(x, y)
            R.output(gv)
        return gv