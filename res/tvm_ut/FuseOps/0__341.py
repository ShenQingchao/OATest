# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def broadcast_to(var_lv1: T.handle, var_T_broadcast_to: T.handle):
        T.func_attr({"op_pattern": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        lv1 = T.match_buffer(var_lv1, (n, n))
        T_broadcast_to = T.match_buffer(var_T_broadcast_to, (T.int64(1), T.int64(1), n, n))
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), n, n):
            with T.block("T_broadcast_to"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv1[v_ax2, v_ax3])
                T.writes(T_broadcast_to[v_ax0, v_ax1, v_ax2, v_ax3])
                T_broadcast_to[v_ax0, v_ax1, v_ax2, v_ax3] = lv1[v_ax2, v_ax3]

    @T.prim_func(private=True)
    def full(var_T_full: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        T_full = T.match_buffer(var_T_full, (n, n))
        # with T.block("root"):
        for ax0, ax1 in T.grid(n, n):
            with T.block("T_full"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads()
                T.writes(T_full[v_ax0, v_ax1])
                T_full[v_ax0, v_ax1] = T.float32(0)

    @T.prim_func(private=True)
    def trilu(var_lv0: T.handle, var_trilu: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        lv0 = T.match_buffer(var_lv0, (n, n))
        trilu = T.match_buffer(var_trilu, (n, n))
        # with T.block("root"):
        for i0, i1 in T.grid(n, n):
            with T.block("trilu"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv0[v_i0, v_i1])
                T.writes(trilu[v_i0, v_i1])
                trilu[v_i0, v_i1] = T.Select(v_i0 < v_i1, lv0[v_i0, v_i1], T.float32(0))

    @R.function
    def main(s: R.Shape(["n"])) -> R.Tensor((1, 1, "n", "n"), dtype="float32"):
        n = T.int64()
        cls = Module
        with R.dataflow():
            lv0 = R.call_tir(cls.full, R.tuple(), out_sinfo=R.Tensor((n, n), dtype="float32"))
            lv1 = R.call_tir(cls.trilu, (lv0,), out_sinfo=R.Tensor((n, n), dtype="float32"))
            gv = R.call_tir(cls.broadcast_to, (lv1,), out_sinfo=R.Tensor((1, 1, n, n), dtype="float32"))
            R.output(gv)
        return gv