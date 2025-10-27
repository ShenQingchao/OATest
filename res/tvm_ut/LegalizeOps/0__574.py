# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main() -> R.Tensor((2, 3), dtype="float32"):
        gv: R.Tensor((2, 3), dtype="float32") = R.ones(R.shape([2, 3]), dtype="float32")
        return gv