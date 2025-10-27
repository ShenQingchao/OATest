# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(lv: T.Buffer((T.int64(16), T.int64(64), T.int64(64)), "float32"), B: T.Buffer((), "float32"), T_add: T.Buffer((T.int64(16), T.int64(64), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(64), T.int64(64)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv[v_ax0, v_ax1, v_ax2], B[()])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = lv[v_ax0, v_ax1, v_ax2] + B[()]

    @T.prim_func(private=True)
    def add1(lv7: T.Buffer((T.int64(16), T.int64(192), T.int64(64)), "float32"), B: T.Buffer((), "float32"), T_add: T.Buffer((T.int64(16), T.int64(192), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(192), T.int64(64)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv7[v_ax0, v_ax1, v_ax2], B[()])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = lv7[v_ax0, v_ax1, v_ax2] + B[()]

    @T.prim_func(private=True)
    def concatenate(lv1: T.Buffer((T.int64(16), T.int64(64), T.int64(64)), "float32"), lv4: T.Buffer((T.int64(16), T.int64(64), T.int64(64)), "float32"), lv5: T.Buffer((T.int64(16), T.int64(64), T.int64(64)), "float32"), T_concat: T.Buffer((T.int64(16), T.int64(192), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(192), T.int64(64)):
            with T.block("T_concat"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv5[v_ax0, v_ax1 - T.int64(128), v_ax2], lv4[v_ax0, v_ax1 - T.int64(64), v_ax2], lv1[v_ax0, v_ax1, v_ax2])
                T.writes(T_concat[v_ax0, v_ax1, v_ax2])
                T_concat[v_ax0, v_ax1, v_ax2] = T.if_then_else(T.int64(128) <= v_ax1, lv5[v_ax0, v_ax1 - T.int64(128), v_ax2], T.if_then_else(T.int64(64) <= v_ax1, lv4[v_ax0, v_ax1 - T.int64(64), v_ax2], lv1[v_ax0, v_ax1, v_ax2]))

    @T.prim_func(private=True)
    def squeeze(x: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float32"), T_squeeze: T.Buffer((T.int64(16), T.int64(64), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(64), T.int64(64)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(x[T.int64(0), v_ax0, v_ax1, v_ax2])
                T.writes(T_squeeze[v_ax0, v_ax1, v_ax2])
                T_squeeze[v_ax0, v_ax1, v_ax2] = x[T.int64(0), v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def squeeze1(lv: T.Buffer((T.int64(16), T.int64(64), T.int64(64)), "float32"), T_squeeze: T.Buffer((T.int64(16), T.int64(64), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(64), T.int64(64)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv[v_ax0, v_ax1, v_ax2])
                T.writes(T_squeeze[v_ax0, v_ax1, v_ax2])
                T_squeeze[v_ax0, v_ax1, v_ax2] = lv[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def squeeze2(lv6: T.Buffer((T.int64(16), T.int64(192), T.int64(64)), "float32"), T_squeeze: T.Buffer((T.int64(16), T.int64(192), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(192), T.int64(64)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv6[v_ax0, v_ax1, v_ax2])
                T.writes(T_squeeze[v_ax0, v_ax1, v_ax2])
                T_squeeze[v_ax0, v_ax1, v_ax2] = lv6[v_ax0, v_ax1, v_ax2]

    @R.function
    def main(x: R.Tensor((1, 16, 64, 64), dtype="float32")) -> R.Tensor((16, 192, 64), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.squeeze, (x,), out_sinfo=R.Tensor((16, 64, 64), dtype="float32"))
            lv1 = R.call_tir(cls.add, (lv, R.const(1, "float32")), out_sinfo=R.Tensor((16, 64, 64), dtype="float32"))
            lv2 = R.call_tir(cls.squeeze1, (lv,), out_sinfo=R.Tensor((16, 64, 64), dtype="float32"))
            lv3 = R.call_tir(cls.add, (lv2, R.const(1, "float32")), out_sinfo=R.Tensor((16, 64, 64), dtype="float32"))
            lv4 = R.call_tir(cls.add, (lv3, R.const(1, "float32")), out_sinfo=R.Tensor((16, 64, 64), dtype="float32"))
            lv5 = R.call_tir(cls.add, (lv, R.const(1, "float32")), out_sinfo=R.Tensor((16, 64, 64), dtype="float32"))
            lv6 = R.call_tir(cls.concatenate, (lv1, lv4, lv5), out_sinfo=R.Tensor((16, 192, 64), dtype="float32"))
            lv7 = R.call_tir(cls.squeeze2, (lv6,), out_sinfo=R.Tensor((16, 192, 64), dtype="float32"))
            gv = R.call_tir(cls.add1, (lv7, R.const(1, "float32")), out_sinfo=R.Tensor((16, 192, 64), dtype="float32"))
            R.output(gv)
        return gv