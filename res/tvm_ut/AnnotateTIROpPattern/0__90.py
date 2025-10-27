# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def sum_sqsum(A: T.Buffer((32, 64), "float32"), vsum: T.Buffer((32,), "float32"), sqsum: T.Buffer((32,), "float32")):
        # with T.block("root"):
        for ax0, k0 in T.grid(32, 64):
            with T.block("block"):
                v_ax0, v_k0 = T.axis.remap("SR", [ax0, k0])
                T.reads(A[v_ax0, v_k0])
                T.writes(vsum[v_ax0], sqsum[v_ax0])
                with T.init():
                    vsum[v_ax0] = T.float32(0)
                    sqsum[v_ax0] = T.float32(0)
                v_vsum: T.float32 = vsum[v_ax0] + A[v_ax0, v_k0]
                v_sqsum: T.float32 = sqsum[v_ax0] + A[v_ax0, v_k0] * A[v_ax0, v_k0]
                vsum[v_ax0] = v_vsum
                sqsum[v_ax0] = v_sqsum