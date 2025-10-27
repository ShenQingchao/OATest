# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="bool"):
        gv: R.Tensor((2, 3), dtype="bool") = R.equal(R.const(1, "float32"), x)
        return gv