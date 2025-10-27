# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def cumsum(var_rxplaceholder: T.handle, out_buf: T.Buffer((160,), "float32")):
        rxplaceholder = T.match_buffer(var_rxplaceholder, (10, 16), offset_factor=1)
        with T.block("cumsum_generic"):
            T.reads(rxplaceholder[0:10, 0:16])
            T.writes(out_buf[0:160])
            for fused in T.parallel(1):
                out_buf[fused * 160] = rxplaceholder[fused * 160 // 16, fused * 160 % 16]
                for v_k in range(159):
                    out_buf[fused * 160 + (v_k + 1)] = out_buf[fused * 160 + (v_k + 1 - 1)] + rxplaceholder[(fused * 160 + (v_k + 1)) // 16, (fused * 160 + (v_k + 1)) % 16]