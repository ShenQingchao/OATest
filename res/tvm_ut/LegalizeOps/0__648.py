# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 2, 3), dtype="float32")) -> R.Tensor((6, 2, 3), dtype="float32"):
        gv: R.Tensor((6, 2, 3), dtype="float32") = R.repeat(x, repeats=2, axis=0)
        return gv