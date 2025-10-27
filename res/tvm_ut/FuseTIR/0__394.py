# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(lv1: T.Buffer((T.int64(10), T.int64(20)), "float32"), lv3: T.Buffer((T.int64(10), T.int64(20)), "float32"), T_add: T.Buffer((T.int64(10), T.int64(20)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(10), T.int64(20)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv1[v_ax0, v_ax1], lv3[v_ax0, v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = lv1[v_ax0, v_ax1] + lv3[v_ax0, v_ax1]

    @R.function
    def fused_add(x: R.Tensor((10, 20), dtype="float32"), y: R.Tensor((10, 20), dtype="float32")) -> R.Tensor((10, 20), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv: R.Tuple(R.Tensor((10, 20), dtype="float32"), R.Tuple(R.Tensor((10, 20), dtype="float32"), R.Tensor((10, 20), dtype="float32"))) = x, (x, y)
            lv1: R.Tensor((10, 20), dtype="float32") = lv[0]
            lv2: R.Tuple(R.Tensor((10, 20), dtype="float32"), R.Tensor((10, 20), dtype="float32")) = lv[1]
            lv3: R.Tensor((10, 20), dtype="float32") = lv2[1]
            gv = R.call_tir(cls.add, (lv1, lv3), out_sinfo=R.Tensor((10, 20), dtype="float32"))
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((10, 20), dtype="float32"), y: R.Tensor((10, 20), dtype="float32")) -> R.Tensor((10, 20), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv1: R.Tensor((10, 20), dtype="float32") = cls.fused_add(x, y)
            R.output(gv1)
        return gv1