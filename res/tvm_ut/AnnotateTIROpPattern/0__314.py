# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(x: T.Buffer((T.int64(10), T.int64(20)), "int32"), p0: T.Buffer((), "int32"), T_add: T.Buffer((T.int64(10), T.int64(20)), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(10), T.int64(20)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(x[v_ax0, v_ax1], p0[()])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = x[v_ax0, v_ax1] + p0[()]

    @T.prim_func(private=True)
    def left_shift(lv1: T.Buffer((T.int64(10), T.int64(20)), "int32"), lv3: T.Buffer((T.int64(10), T.int64(20)), "int32"), T_left_shift: T.Buffer((T.int64(10), T.int64(20)), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(10), T.int64(20)):
            with T.block("T_left_shift"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv1[v_ax0, v_ax1], lv3[v_ax0, v_ax1])
                T.writes(T_left_shift[v_ax0, v_ax1])
                T_left_shift[v_ax0, v_ax1] = T.shift_left(lv1[v_ax0, v_ax1], lv3[v_ax0, v_ax1])

    @T.prim_func(private=True)
    def squeeze(lv: T.Buffer((T.int64(10), T.int64(20)), "int32"), T_squeeze: T.Buffer((T.int64(10), T.int64(20)), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(10), T.int64(20)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv[v_ax0, v_ax1])
                T.writes(T_squeeze[v_ax0, v_ax1])
                T_squeeze[v_ax0, v_ax1] = lv[v_ax0, v_ax1]

    @T.prim_func(private=True)
    def transpose(lv: T.Buffer((T.int64(10), T.int64(20)), "int32"), T_transpose: T.Buffer((T.int64(20), T.int64(10)), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(20), T.int64(10)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = lv[v_ax1, v_ax0]

    @T.prim_func(private=True)
    def transpose1(lv2: T.Buffer((T.int64(20), T.int64(10)), "int32"), T_transpose: T.Buffer((T.int64(10), T.int64(20)), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(10), T.int64(20)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv2[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = lv2[v_ax1, v_ax0]

    @R.function(private=True)
    def fused_add_squeeze_transpose_transpose1_left_shift(x: R.Tensor((10, 20), dtype="int32"), p0: R.Tensor((), dtype="int32")) -> R.Tensor((10, 20), dtype="int32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.add, (x, p0), out_sinfo=R.Tensor((10, 20), dtype="int32"))
            lv1 = R.call_tir(cls.squeeze, (lv,), out_sinfo=R.Tensor((10, 20), dtype="int32"))
            lv2 = R.call_tir(cls.transpose, (lv,), out_sinfo=R.Tensor((20, 10), dtype="int32"))
            lv3 = R.call_tir(cls.transpose1, (lv2,), out_sinfo=R.Tensor((10, 20), dtype="int32"))
            gv = R.call_tir(cls.left_shift, (lv1, lv3), out_sinfo=R.Tensor((10, 20), dtype="int32"))
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((10, 20), dtype="int32")) -> R.Tensor((10, 20), dtype="int32"):
        cls = Module
        with R.dataflow():
            gv1: R.Tensor((10, 20), dtype="int32") = cls.fused_add_squeeze_transpose_transpose1_left_shift(x, R.const(1, "int32"))
            R.output(gv1)
        return gv1