# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(x: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float32"), p0: T.Buffer((), "float32"), T_add: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16), T.int64(64), T.int64(64)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(x[v_ax0, v_ax1, v_ax2, v_ax3], p0[()])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = x[v_ax0, v_ax1, v_ax2, v_ax3] + p0[()]

    @T.prim_func(private=True)
    def add1(lv3: T.Buffer((T.int64(1), T.int64(48), T.int64(64), T.int64(64)), "float32"), p3: T.Buffer((), "float32"), T_add: T.Buffer((T.int64(1), T.int64(48), T.int64(64), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(48), T.int64(64), T.int64(64)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv3[v_ax0, v_ax1, v_ax2, v_ax3], p3[()])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = lv3[v_ax0, v_ax1, v_ax2, v_ax3] + p3[()]

    @T.prim_func(private=True)
    def add2(lv15: T.Buffer((T.int64(1), T.int64(144), T.int64(32), T.int64(32)), "float32"), p0: T.Buffer((), "float32"), T_add: T.Buffer((T.int64(1), T.int64(144), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(144), T.int64(32), T.int64(32)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv15[v_ax0, v_ax1, v_ax2, v_ax3], p0[()])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = lv15[v_ax0, v_ax1, v_ax2, v_ax3] + p0[()]

    @T.prim_func(private=True)
    def concatenate(lv: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float32"), lv1: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float32"), lv2: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float32"), T_concat: T.Buffer((T.int64(1), T.int64(48), T.int64(64), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(48), T.int64(64), T.int64(64)):
            with T.block("T_concat"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv2[v_ax0, v_ax1 - T.int64(32), v_ax2, v_ax3], lv1[v_ax0, v_ax1 - T.int64(16), v_ax2, v_ax3], lv[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_concat[v_ax0, v_ax1, v_ax2, v_ax3])
                T_concat[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(32) <= v_ax1, lv2[v_ax0, v_ax1 - T.int64(32), v_ax2, v_ax3], T.if_then_else(T.int64(16) <= v_ax1, lv1[v_ax0, v_ax1 - T.int64(16), v_ax2, v_ax3], lv[v_ax0, v_ax1, v_ax2, v_ax3]))

    @T.prim_func(private=True)
    def concatenate1(lv4: T.Buffer((T.int64(1), T.int64(48), T.int64(64), T.int64(64)), "float32"), lv9: T.Buffer((T.int64(1), T.int64(48), T.int64(64), T.int64(64)), "float32"), lv14: T.Buffer((T.int64(1), T.int64(48), T.int64(64), T.int64(64)), "float32"), T_concat: T.Buffer((T.int64(1), T.int64(144), T.int64(64), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(144), T.int64(64), T.int64(64)):
            with T.block("T_concat"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv14[v_ax0, v_ax1 - T.int64(96), v_ax2, v_ax3], lv9[v_ax0, v_ax1 - T.int64(48), v_ax2, v_ax3], lv4[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_concat[v_ax0, v_ax1, v_ax2, v_ax3])
                T_concat[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(96) <= v_ax1, lv14[v_ax0, v_ax1 - T.int64(96), v_ax2, v_ax3], T.if_then_else(T.int64(48) <= v_ax1, lv9[v_ax0, v_ax1 - T.int64(48), v_ax2, v_ax3], lv4[v_ax0, v_ax1, v_ax2, v_ax3]))

    @T.prim_func(private=True)
    def pool2d(concat: T.Buffer((T.int64(1), T.int64(144), T.int64(64), T.int64(64)), "float32"), pool_max: T.Buffer((T.int64(1), T.int64(144), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, rv0, rv1 in T.grid(T.int64(1), T.int64(144), T.int64(32), T.int64(32), T.int64(2), T.int64(2)):
            with T.block("pool_max"):
                v_ax0, v_ax1, v_ax2, v_ax3, v_rv0, v_rv1 = T.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, rv0, rv1])
                T.reads(concat[v_ax0, v_ax1, v_ax2 * T.int64(2) + v_rv0, v_ax3 * T.int64(2) + v_rv1])
                T.writes(pool_max[v_ax0, v_ax1, v_ax2, v_ax3])
                T.block_attr({"schedule_rule": "meta_schedule.pool_max"})
                with T.init():
                    pool_max[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(-3.4028234663852886e+38)
                pool_max[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(pool_max[v_ax0, v_ax1, v_ax2, v_ax3], concat[v_ax0, v_ax1, v_ax2 * T.int64(2) + v_rv0, v_ax3 * T.int64(2) + v_rv1])

    @R.function(private=True)
    def fused_add_add_add_concatenate_add1_add_add_add_concatenate_add1_add_add_add_concatenate_add1_concatenate1(x: R.Tensor((1, 16, 64, 64), dtype="float32"), p0: R.Tensor((), dtype="float32"), p1: R.Tensor((), dtype="float32"), p2: R.Tensor((), dtype="float32"), p3: R.Tensor((), dtype="float32"), p4: R.Tensor((), dtype="float32"), p5: R.Tensor((), dtype="float32"), p6: R.Tensor((), dtype="float32"), p7: R.Tensor((), dtype="float32"), p8: R.Tensor((), dtype="float32"), p9: R.Tensor((), dtype="float32"), p10: R.Tensor((), dtype="float32"), p11: R.Tensor((), dtype="float32")) -> R.Tensor((1, 144, 64, 64), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.add, (x, p0), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float32"))
            lv1 = R.call_tir(cls.add, (x, p1), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float32"))
            lv2 = R.call_tir(cls.add, (x, p2), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float32"))
            lv3 = R.call_tir(cls.concatenate, (lv, lv1, lv2), out_sinfo=R.Tensor((1, 48, 64, 64), dtype="float32"))
            lv4 = R.call_tir(cls.add1, (lv3, p3), out_sinfo=R.Tensor((1, 48, 64, 64), dtype="float32"))
            lv5 = R.call_tir(cls.add, (x, p4), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float32"))
            lv6 = R.call_tir(cls.add, (x, p5), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float32"))
            lv7 = R.call_tir(cls.add, (x, p6), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float32"))
            lv8 = R.call_tir(cls.concatenate, (lv5, lv6, lv7), out_sinfo=R.Tensor((1, 48, 64, 64), dtype="float32"))
            lv9 = R.call_tir(cls.add1, (lv8, p7), out_sinfo=R.Tensor((1, 48, 64, 64), dtype="float32"))
            lv10 = R.call_tir(cls.add, (x, p8), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float32"))
            lv11 = R.call_tir(cls.add, (x, p9), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float32"))
            lv12 = R.call_tir(cls.add, (x, p10), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float32"))
            lv13 = R.call_tir(cls.concatenate, (lv10, lv11, lv12), out_sinfo=R.Tensor((1, 48, 64, 64), dtype="float32"))
            lv14 = R.call_tir(cls.add1, (lv13, p11), out_sinfo=R.Tensor((1, 48, 64, 64), dtype="float32"))
            gv = R.call_tir(cls.concatenate1, (lv4, lv9, lv14), out_sinfo=R.Tensor((1, 144, 64, 64), dtype="float32"))
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_pool2d_add2(concat: R.Tensor((1, 144, 64, 64), dtype="float32"), p0: R.Tensor((), dtype="float32")) -> R.Tensor((1, 144, 32, 32), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv15 = R.call_tir(cls.pool2d, (concat,), out_sinfo=R.Tensor((1, 144, 32, 32), dtype="float32"))
            gv1 = R.call_tir(cls.add2, (lv15, p0), out_sinfo=R.Tensor((1, 144, 32, 32), dtype="float32"))
            R.output(gv1)
        return gv1

    @R.function
    def main(x: R.Tensor((1, 16, 64, 64), dtype="float32")) -> R.Tuple(R.Tensor((1, 144, 32, 32), dtype="float32"), R.Tensor((1, 144, 32, 32), dtype="float32")):
        cls = Module
        with R.dataflow():
            lv16: R.Tensor((1, 144, 64, 64), dtype="float32") = cls.fused_add_add_add_concatenate_add1_add_add_add_concatenate_add1_add_add_add_concatenate_add1_concatenate1(x, R.const(1, "float32"), R.const(1, "float32"), R.const(1, "float32"), R.const(1, "float32"), R.const(1, "float32"), R.const(1, "float32"), R.const(1, "float32"), R.const(1, "float32"), R.const(1, "float32"), R.const(1, "float32"), R.const(1, "float32"), R.const(1, "float32"))
            lv17: R.Tensor((1, 144, 32, 32), dtype="float32") = cls.fused_pool2d_add2(lv16, R.const(1, "float32"))
            lv18 = R.call_tir(cls.add2, (lv17, R.const(1, "float32")), out_sinfo=R.Tensor((1, 144, 32, 32), dtype="float32"))
            gv2: R.Tuple(R.Tensor((1, 144, 32, 32), dtype="float32"), R.Tensor((1, 144, 32, 32), dtype="float32")) = lv17, lv18
            R.output(gv2)
        return gv2