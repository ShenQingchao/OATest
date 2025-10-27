# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 2, 3), dtype="float32")) -> R.Tensor((2, 1), dtype="float32"):
        gv: R.Tensor((2, 1), dtype="float32") = R.collapse_sum_to(x, R.shape([2, 1]))
        return gv