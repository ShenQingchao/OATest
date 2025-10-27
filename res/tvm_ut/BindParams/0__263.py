# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(c0: R.Tensor((16, 16), dtype="float32")) -> R.Tensor((16, 16), dtype="float32"):
        with R.dataflow():
            gv: R.Tensor((16, 16), dtype="float32") = R.add(c0, c0)
            R.output(gv)
        return gv