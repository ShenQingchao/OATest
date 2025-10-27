# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(condition: R.Tensor((3, 2, 1), dtype="bool"), x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 1), dtype="float32")) -> R.Tensor((3, 2, 3), dtype="float32"):
        gv: R.Tensor((3, 2, 3), dtype="float32") = R.where(condition, x, y)
        return gv