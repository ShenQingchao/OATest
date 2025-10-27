# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((2, 4, 3, 1), dtype="float32"):
        gv: R.Tensor((2, 4, 3, 1), dtype="float32") = R.permute_dims(x, axes=[1, -1, 2, -4])
        return gv