# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def exp(x1: T.Buffer((T.int64(10), T.int64(20)), "float32"), compute: T.Buffer((T.int64(10), T.int64(20)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(10), T.int64(20)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(x1[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.exp(x1[v_i0, v_i1])

    @T.prim_func(private=True)
    def squeeze(lv: T.Buffer((T.int64(10), T.int64(20)), "float32"), T_squeeze: T.Buffer((T.int64(10), T.int64(20)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(10), T.int64(20)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv[v_ax0, v_ax1])
                T.writes(T_squeeze[v_ax0, v_ax1])
                T_squeeze[v_ax0, v_ax1] = lv[v_ax0, v_ax1]

    @R.function
    def fused_exp_squeeze(x1: R.Tensor((10, 20), dtype="float32")) -> R.Tensor((10, 20), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.exp, (x1,), out_sinfo=R.Tensor((10, 20), dtype="float32"))
            gv = R.call_tir(cls.squeeze, (lv,), out_sinfo=R.Tensor((10, 20), dtype="float32"))
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((10, 20), dtype="float32")) -> R.Tensor((10, 20), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv1: R.Tensor((10, 20), dtype="float32") = cls.fused_exp_squeeze(x)
            lv2: R.Tensor((10, 20), dtype="float32") = cls.fused_exp_squeeze(lv1)
            gv1: R.Tensor((10, 20), dtype="float32") = lv2
            R.output(gv1)
        return gv1