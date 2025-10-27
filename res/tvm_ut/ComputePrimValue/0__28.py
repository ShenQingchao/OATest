# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(_N: R.Prim(value="N"), _M: R.Prim(value="M")) -> R.Prim(value="N * M"):
        N = T.int64()
        M = T.int64()
        out: R.Prim(value=N * M) = R.prim_value(N * M)
        return out