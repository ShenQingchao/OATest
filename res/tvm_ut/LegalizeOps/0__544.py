# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1, 2, 3), dtype="float32"), y: R.Tensor((4, 3, 2, 1), dtype="float32")) -> R.Tensor((4, 3, 2, 3), dtype="bool"):
        gv: R.Tensor((4, 3, 2, 3), dtype="bool") = R.less_equal(x, y)
        return gv