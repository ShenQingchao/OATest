# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(pure=False)
    def main(A: R.Tensor(("N",))) -> R.Tensor(("N",)):
        N = T.int64()
        R.assert_op(R.prim_value(N % 16 == 0), format=R.str(""))
        return A