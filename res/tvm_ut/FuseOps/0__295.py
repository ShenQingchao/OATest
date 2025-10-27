# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def pool2d(x: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float32"), pool_max: T.Buffer((T.int64(1), T.int64(16), T.int64(32), T.int64(32)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, rv0, rv1 in T.grid(T.int64(1), T.int64(16), T.int64(32), T.int64(32), T.int64(2), T.int64(2)):
            with T.block("pool_max"):
                v_ax0, v_ax1, v_ax2, v_ax3, v_rv0, v_rv1 = T.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, rv0, rv1])
                T.reads(x[v_ax0, v_ax1, v_ax2 * T.int64(2) + v_rv0, v_ax3 * T.int64(2) + v_rv1])
                T.writes(pool_max[v_ax0, v_ax1, v_ax2, v_ax3])
                T.block_attr({"schedule_rule": "meta_schedule.pool_max"})
                with T.init():
                    pool_max[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(-3.4028234663852886e+38)
                pool_max[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(pool_max[v_ax0, v_ax1, v_ax2, v_ax3], x[v_ax0, v_ax1, v_ax2 * T.int64(2) + v_rv0, v_ax3 * T.int64(2) + v_rv1])

    @T.prim_func(private=True)
    def upsampling(lv: T.Buffer((T.int64(1), T.int64(16), T.int64(32), T.int64(32)), "float32"), resize: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(16), T.int64(64), T.int64(64)):
            with T.block("resize"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(lv[v_i0, v_i1, T.max(T.min(T.Div(v_i2, T.int64(2)), T.int64(31)), T.int64(0)), T.max(T.min(T.Div(v_i3, T.int64(2)), T.int64(31)), T.int64(0))])
                T.writes(resize[v_i0, v_i1, v_i2, v_i3])
                resize[v_i0, v_i1, v_i2, v_i3] = lv[v_i0, v_i1, T.max(T.min(T.Div(v_i2, T.int64(2)), T.int64(31)), T.int64(0)), T.max(T.min(T.Div(v_i3, T.int64(2)), T.int64(31)), T.int64(0))]

    @R.function
    def main(x: R.Tensor((1, 16, 64, 64), dtype="float32")) -> R.Tuple(R.Tensor((1, 16, 64, 64), dtype="float32"), R.Tensor((1, 16, 64, 64), dtype="float32")):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.pool2d, (x,), out_sinfo=R.Tensor((1, 16, 32, 32), dtype="float32"))
            lv1 = R.call_tir(cls.upsampling, (lv,), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float32"))
            gv: R.Tuple(R.Tensor((1, 16, 64, 64), dtype="float32"), R.Tensor((1, 16, 64, 64), dtype="float32")) = lv1, x
            R.output(gv)
        return gv