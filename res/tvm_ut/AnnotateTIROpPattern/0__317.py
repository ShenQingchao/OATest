# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def cast(lv: T.Buffer((T.int64(16), T.int64(16)), "float32"), compute: T.Buffer((T.int64(16), T.int64(16)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(16), T.int64(16)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", lv[v_i0, v_i1])

    @T.prim_func(private=True)
    def softmax(x: T.Buffer((T.int64(16), T.int64(16)), "float32"), T_softmax_norm: T.Buffer((T.int64(16), T.int64(16)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(16),))
        T_softmax_exp = T.alloc_buffer((T.int64(16), T.int64(16)))
        T_softmax_expsum = T.alloc_buffer((T.int64(16),))
        for i0, k in T.grid(T.int64(16), T.int64(16)):
            with T.block("T_softmax_maxelem"):
                v_i0, v_k = T.axis.remap("SR", [i0, k])
                T.reads(x[v_i0, v_k])
                T.writes(T_softmax_maxelem[v_i0])
                with T.init():
                    T_softmax_maxelem[v_i0] = T.float32(-3.4028234663852886e+38)
                T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], x[v_i0, v_k])
        for i0, i1 in T.grid(T.int64(16), T.int64(16)):
            with T.block("T_softmax_exp"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(x[v_i0, v_i1], T_softmax_maxelem[v_i0])
                T.writes(T_softmax_exp[v_i0, v_i1])
                T_softmax_exp[v_i0, v_i1] = T.exp(x[v_i0, v_i1] - T_softmax_maxelem[v_i0])
        for i0, k in T.grid(T.int64(16), T.int64(16)):
            with T.block("T_softmax_expsum"):
                v_i0, v_k = T.axis.remap("SR", [i0, k])
                T.reads(T_softmax_exp[v_i0, v_k])
                T.writes(T_softmax_expsum[v_i0])
                with T.init():
                    T_softmax_expsum[v_i0] = T.float32(0)
                T_softmax_expsum[v_i0] = T_softmax_expsum[v_i0] + T_softmax_exp[v_i0, v_k]
        for i0, i1 in T.grid(T.int64(16), T.int64(16)):
            with T.block("T_softmax_norm"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_softmax_exp[v_i0, v_i1], T_softmax_expsum[v_i0])
                T.writes(T_softmax_norm[v_i0, v_i1])
                T.block_attr({"axis": 1})
                T_softmax_norm[v_i0, v_i1] = T_softmax_exp[v_i0, v_i1] / T_softmax_expsum[v_i0]

    @R.function(private=True)
    def fused_softmax_cast(x: R.Tensor((16, 16), dtype="float32")) -> R.Tensor((16, 16), dtype="float16"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.softmax, (x,), out_sinfo=R.Tensor((16, 16), dtype="float32"))
            gv = R.call_tir(cls.cast, (lv,), out_sinfo=R.Tensor((16, 16), dtype="float16"))
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((16, 16), dtype="float32")) -> R.Tensor((16, 16), dtype="float16"):
        cls = Module
        with R.dataflow():
            gv1: R.Tensor((16, 16), dtype="float16") = cls.fused_softmax_cast(x)
            R.output(gv1)
        return gv1