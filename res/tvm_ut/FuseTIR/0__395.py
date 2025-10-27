# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(lv1: T.Buffer((T.int64(10),), "int64"), offset: T.Buffer((T.int64(10),), "int32"), T_add: T.Buffer((T.int64(10),), "int64")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0 in range(T.int64(10)):
            with T.block("T_add"):
                v_ax0 = T.axis.spatial(T.int64(10), ax0)
                T.reads(lv1[v_ax0], offset[v_ax0])
                T.writes(T_add[v_ax0])
                T_add[v_ax0] = lv1[v_ax0] + T.Cast("int64", offset[v_ax0])

    @T.prim_func(private=True)
    def te_argmax_idx_val(x: T.Buffer((T.int64(10), T.int64(20)), "float32"), argmax_v0: T.Buffer((T.int64(10),), "int64"), argmax_v1: T.Buffer((T.int64(10),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, k in T.grid(T.int64(10), T.int64(20)):
            with T.block("argmax"):
                v_i, v_k = T.axis.remap("SR", [i, k])
                T.reads(x[v_i, v_k])
                T.writes(argmax_v0[v_i], argmax_v1[v_i])
                with T.init():
                    argmax_v0[v_i] = T.int64(-1)
                    argmax_v1[v_i] = T.float32(-3.4028234663852886e+38)
                v_argmax_v0: T.int64 = T.Select(argmax_v1[v_i] >= x[v_i, v_k], argmax_v0[v_i], v_k)
                v_argmax_v1: T.float32 = T.Select(argmax_v1[v_i] >= x[v_i, v_k], argmax_v1[v_i], x[v_i, v_k])
                argmax_v0[v_i] = v_argmax_v0
                argmax_v1[v_i] = v_argmax_v1

    @R.function
    def fused_argmax_add(x: R.Tensor((10, 20), dtype="float32"), offset: R.Tensor((10,), dtype="int32")) -> R.Tensor((10,), dtype="int64"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.te_argmax_idx_val, (x,), out_sinfo=[R.Tensor((10,), dtype="int64"), R.Tensor((10,), dtype="float32")])
            lv1: R.Tensor((10,), dtype="int64") = lv[0]
            gv = R.call_tir(cls.add, (lv1, offset), out_sinfo=R.Tensor((10,), dtype="int64"))
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((10, 20), dtype="float32"), x_1: R.Tensor((10,), dtype="int32")) -> R.Tensor((10,), dtype="int64"):
        cls = Module
        with R.dataflow():
            gv1: R.Tensor((10,), dtype="int64") = cls.fused_argmax_add(x, x_1)
            R.output(gv1)
        return gv1