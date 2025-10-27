# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((), dtype="float32")) -> R.Tensor((1,), dtype="float32"):
        gv: R.Tensor((1,), dtype="float32") = R.flatten(x)
        return gv