# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3), dtype="int32")) -> R.Tensor((2, 3), dtype="float32"):
        gv: R.Tensor((2, 3), dtype="float32") = R.full_like(x, R.const(-5, "float32"), dtype="void")
        return gv