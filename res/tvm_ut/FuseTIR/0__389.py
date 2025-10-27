# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(lv2: T.Buffer((T.int64(10),), "float32"), lv1: T.Buffer((T.int64(10),), "float32"), T_add: T.Buffer((T.int64(10),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0 in range(T.int64(10)):
            with T.block("T_add"):
                v_ax0 = T.axis.spatial(T.int64(10), ax0)
                T.reads(lv2[v_ax0], lv1[v_ax0])
                T.writes(T_add[v_ax0])
                T_add[v_ax0] = lv2[v_ax0] + lv1[v_ax0]

    @T.prim_func(private=True)
    def exp(lv: T.Buffer((T.int64(10),), "float32"), compute: T.Buffer((T.int64(10),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0 in range(T.int64(10)):
            with T.block("compute"):
                v_i0 = T.axis.spatial(T.int64(10), i0)
                T.reads(lv[v_i0])
                T.writes(compute[v_i0])
                compute[v_i0] = T.exp(lv[v_i0])

    @R.function(private=True)
    def fused_exp_add(x: R.Tuple(R.Tensor((10,), dtype="float32"), R.Tensor((10,), dtype="float32"))) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv: R.Tensor((10,), dtype="float32") = x[0]
            lv1: R.Tensor((10,), dtype="float32") = x[1]
            lv2 = R.call_tir(cls.exp, (lv,), out_sinfo=R.Tensor((10,), dtype="float32"))
            gv = R.call_tir(cls.add, (lv2, lv1), out_sinfo=R.Tensor((10,), dtype="float32"))
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tuple(R.Tensor((10,), dtype="float32"), R.Tensor((10,), dtype="float32"))) -> R.Tensor((10,), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv1: R.Tensor((10,), dtype="float32") = cls.fused_exp_add(x)
            R.output(gv1)
        return gv1