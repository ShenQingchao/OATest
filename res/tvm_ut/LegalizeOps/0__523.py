# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def power(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_T_power: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        c, d = T.int64(), T.int64()
        rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(1), c, d))
        a, b = T.int64(), T.int64()
        rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (a, b, c, T.int64(1)))
        T_power = T.match_buffer(var_T_power, (a, b, c, d))
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(a, b, c, d):
            with T.block("T_power"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(rxplaceholder[T.int64(0), v_ax2, v_ax3], rxplaceholder_1[v_ax0, v_ax1, v_ax2, T.int64(0)])
                T.writes(T_power[v_ax0, v_ax1, v_ax2, v_ax3])
                T_power[v_ax0, v_ax1, v_ax2, v_ax3] = T.pow(rxplaceholder[T.int64(0), v_ax2, v_ax3], rxplaceholder_1[v_ax0, v_ax1, v_ax2, T.int64(0)])

    @R.function
    def main(x: R.Tensor((1, "c", "d"), dtype="float32"), y: R.Tensor(("a", "b", "c", 1), dtype="float32")) -> R.Tensor(("a", "b", "c", "d"), dtype="float32"):
        a = T.int64()
        b = T.int64()
        c = T.int64()
        d = T.int64()
        cls = Module
        gv = R.call_tir(cls.power, (x, y), out_sinfo=R.Tensor((a, b, c, d), dtype="float32"))
        return gv