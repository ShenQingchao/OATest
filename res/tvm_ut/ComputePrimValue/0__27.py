# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(pure=False)
    def main(A: R.Tensor(("N",))) -> R.Tensor(ndim=1):
        N = T.int64()
        if R.prim_value(N % 16 == 0):
            out_then: R.Tensor((N,)) = R.call_packed("fast_vectorized_impl", A, sinfo_args=(R.Tensor((N,)),))
            out: R.Tensor(ndim=1) = out_then
        else:
            out_else: R.Tensor((N,)) = R.call_packed("slow_non_vectorized_impl", A, sinfo_args=(R.Tensor((N,)),))
            out: R.Tensor(ndim=1) = out_else
        return out