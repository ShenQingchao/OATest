# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 1, 3), dtype="float32")) -> R.Tensor((4, 2, 5, 3), dtype="float32"):
        gv: R.Tensor((4, 2, 5, 3), dtype="float32") = R.broadcast_to(x, R.shape([4, 2, 5, 3]))
        return gv