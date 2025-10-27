# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main() -> R.Tensor((), dtype="int32"):
        gv: R.Tensor((), dtype="int32") = R.astype(R.const(1.5, "float32"), dtype="int32")
        return gv