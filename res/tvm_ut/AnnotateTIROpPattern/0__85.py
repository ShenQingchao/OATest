# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def add_with_unit_dim_len_broadcast(A: T.Buffer((1, 64, 112, 112), "float32"), B: T.Buffer((64, 1, 1), "float32"), C: T.Buffer((1, 64, 112, 112), "float32")):
        T.func_attr({"global_symbol": "add5", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(1, 64, 112, 112):
            with T.block("T_add"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(A[ax0, ax1, ax2, ax3], B[ax1, 0, 0])
                T.writes(C[ax0, ax1, ax2, ax3])
                C[ax0, ax1, ax2, ax3] = A[ax0, ax1, ax2, ax3] + B[ax1, 0, 0]