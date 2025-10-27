# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x0: R.Tensor((3, 3), dtype="float32"), x1: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((3, 3), dtype="float32"):
        with R.dataflow():
            gv: R.Tensor((3, 3), dtype="float32") = R.add(x0, x1)
            R.output(gv)
        return gv