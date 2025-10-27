# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="int32"):
        with R.dataflow():
            gv: R.Tensor((), dtype="int32") = R.const(1, "int32")
            R.output(gv)
        return gv