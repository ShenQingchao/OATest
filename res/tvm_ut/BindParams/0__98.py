# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(A: R.Tensor((16,), dtype="float32")) -> R.Tensor((16,), dtype="float32"):
        return A