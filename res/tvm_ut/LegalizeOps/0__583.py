# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("n",), dtype="float32")) -> R.Tensor(("n // 2",), dtype="int64"):
        n = T.int64()
        gv: R.Tensor((n // 2,), dtype="int64") = R.arange(R.prim_value(1), R.prim_value(n), R.prim_value(2), dtype="int64")
        return gv