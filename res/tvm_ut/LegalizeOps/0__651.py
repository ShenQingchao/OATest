# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 2, 3), dtype="float32")) -> R.Tensor((2, 3, 4, 9), dtype="float32"):
        gv: R.Tensor((2, 3, 4, 9), dtype="float32") = R.tile(x, repeats=[2, 1, 2, 3])
        return gv