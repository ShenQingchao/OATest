# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(x: T.Buffer((T.int64(10), T.int64(20)), "float32"), p0: T.Buffer((), "float32"), T_add: T.Buffer((T.int64(10), T.int64(20)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(10), T.int64(20)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(x[v_ax0, v_ax1], p0[()])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = x[v_ax0, v_ax1] + p0[()]

    @T.prim_func(private=True)
    def exp(gv: T.Buffer((T.int64(10), T.int64(20)), "float32"), compute: T.Buffer((T.int64(10), T.int64(20)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(10), T.int64(20)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(gv[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.exp(gv[v_i0, v_i1])

    @R.function
    def fused_add_exp(x: R.Tensor((10, 20), dtype="float32"), p0: R.Tensor((), dtype="float32")) -> R.Tuple(R.Tensor((10, 20), dtype="float32"), R.Tensor((10, 20), dtype="float32")):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.add, (x, p0), out_sinfo=R.Tensor((10, 20), dtype="float32"))
            gv1 = R.call_tir(cls.exp, (gv,), out_sinfo=R.Tensor((10, 20), dtype="float32"))
            R.output(gv, gv1)
        return (gv, gv1)

    @R.function
    def main(x: R.Tensor((10, 20), dtype="float32"), p0: R.Tensor((), dtype="float32")) -> R.Tuple(R.Tensor((10, 20), dtype="float32"), R.Tensor((10, 20), dtype="float32")):
        cls = Module
        with R.dataflow():
            gv2: R.Tuple(R.Tensor((10, 20), dtype="float32"), R.Tensor((10, 20), dtype="float32")) = cls.fused_add_exp(x, p0)
            R.output(gv2)
        return gv2