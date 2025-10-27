# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(lv1: T.Buffer((T.int64(1),), "float32"), lv1_1: T.Buffer((T.int64(1),), "float32"), T_add: T.Buffer((T.int64(1),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0 in range(T.int64(1)):
            with T.block("T_add"):
                v_ax0 = T.axis.spatial(T.int64(1), ax0)
                T.reads(lv1[v_ax0], lv1_1[v_ax0])
                T.writes(T_add[v_ax0])
                T_add[v_ax0] = lv1[v_ax0] + lv1_1[v_ax0]

    @T.prim_func(private=True)
    def reshape(x: T.Buffer((), "float32"), T_reshape: T.Buffer((T.int64(1),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0 in range(T.int64(1)):
            with T.block("T_reshape"):
                v_ax0 = T.axis.spatial(T.int64(1), ax0)
                T.reads(x[()])
                T.writes(T_reshape[v_ax0])
                T_reshape[v_ax0] = x[()]

    @R.function
    def main(x: R.Tensor((), dtype="float32")) -> R.Tensor((1,), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv1 = R.call_tir(cls.reshape, (x,), out_sinfo=R.Tensor((1,), dtype="float32"))
            lv2 = R.call_tir(cls.add, (lv1, lv1), out_sinfo=R.Tensor((1,), dtype="float32"))
            R.output(lv2)
        return lv2