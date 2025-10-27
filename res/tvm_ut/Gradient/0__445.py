# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x0: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
        gv: R.Tensor((), dtype="float32") = R.sum(x0, axis=None, keepdims=False)
        return gv