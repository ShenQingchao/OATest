# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x1: R.Tensor((1, 2, 3), dtype="float32"), x2: R.Tensor((1, 3, 3), dtype="float32"), x3: R.Tensor((1, 4, 3), dtype="float32")) -> R.Tensor((1, 9, 3), dtype="float32"):
        gv: R.Tensor((1, 9, 3), dtype="float32") = R.concat((x1, x2, x3), axis=1)
        return gv