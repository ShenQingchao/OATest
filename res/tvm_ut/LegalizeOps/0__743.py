# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3, 4, 5), dtype="float32")) -> R.Tensor((), dtype="float32"):
        gv: R.Tensor((), dtype="float32") = R.std(x, axis=None, keepdims=False)
        return gv