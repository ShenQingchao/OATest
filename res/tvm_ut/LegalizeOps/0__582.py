# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main() -> R.Tensor((5,), dtype="int64"):
        gv: R.Tensor((5,), dtype="int64") = R.arange(R.prim_value(1), R.prim_value(10), R.prim_value(2), dtype="int64")
        return gv