# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 3), dtype="float16")) -> R.Tensor((3, 3), dtype="float16"):
        gv: R.Tensor((3, 3), dtype="float16") = R.multiply(x, R.const(1.1455078125, "float16"))
        return gv