# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def no_buffer_stores(A: T.Buffer((32, 64), "float32"), vsum: T.Buffer((32,), "float32")):
        # with T.block("root"):
        for ax0, k0 in T.grid(32, 64):
            with T.block("block"):
                v_ax0, v_k0 = T.axis.remap("SR", [ax0, k0])
                T.reads(A[v_ax0, v_k0])
                T.writes(vsum[v_ax0])
                T.call_packed("some_func")