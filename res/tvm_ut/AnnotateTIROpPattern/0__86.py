# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def add_zero_dim(A: T.Buffer((128,), "float32"), B: T.Buffer((), "float32"), C: T.Buffer((128,), "float32")):
        T.func_attr({"global_symbol": "add8", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0 in range(128):
            with T.block("T_add"):
                ax0 = T.axis.spatial(128, i0)
                T.reads(A[ax0], B[()])
                T.writes(C[ax0])
                C[ax0] = A[ax0] + B[()]