# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3, 4), dtype="float32")) -> R.Tensor((2, 1, 1, 1, 3, 1, 4, 1), dtype="float32"):
        gv: R.Tensor((2, 1, 1, 1, 3, 1, 4, 1), dtype="float32") = R.expand_dims(x, axis=[-1, 1, -6, 3, 5])
        return gv