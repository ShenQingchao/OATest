# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((1, 3), dtype="float32")) -> R.Tensor((1, 3), dtype="float32"):
        gv: R.Tensor((1, 3), dtype="float32") = R.collapse_sum_like(x, y)
        return gv