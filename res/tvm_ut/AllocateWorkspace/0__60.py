# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def entry_a(q: R.Tensor((32, 8, 16, 8), dtype="float16"), k: R.Tensor((32, 8, 16, 8), dtype="float16"), v: R.Tensor((32, 8, 16, 8), dtype="float16")) -> R.Tensor((32, 8, 16, 8), dtype="float16"):
        cls = Module
        with R.dataflow():
            gv: R.Tensor((32, 8, 16, 8), dtype="float16") = cls.fused_relax_nn_attention_cutlass(q, k, v)
            R.output(gv)
        return gv

    @R.function
    def entry_b(q: R.Tensor((32, 8, 16, 8), dtype="float16"), k: R.Tensor((32, 8, 16, 8), dtype="float16"), v: R.Tensor((32, 8, 16, 8), dtype="float16")) -> R.Tensor((32, 8, 16, 8), dtype="float16"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((32, 8, 16, 8), dtype="float16") = cls.fused_relax_nn_attention_cutlass(q, k, v)
            gv: R.Tensor((32, 8, 16, 8), dtype="float16") = R.add(lv, R.const(1, "float16"))
            R.output(gv)
        return gv

    @R.function
    def fused_relax_nn_attention_cutlass(q: R.Tensor((32, 8, 16, 8), dtype="float16"), k: R.Tensor((32, 8, 16, 8), dtype="float16"), v: R.Tensor((32, 8, 16, 8), dtype="float16")) -> R.Tensor((32, 8, 16, 8), dtype="float16"):
        R.func_attr({"Codegen": "cutlass", "WorkspaceSize": 65536})
        # from tvm.script import relax as R
        
        @R.function
        def gv(q_1: R.Tensor((32, 8, 16, 8), dtype="float16"), k_1: R.Tensor((32, 8, 16, 8), dtype="float16"), v_1: R.Tensor((32, 8, 16, 8), dtype="float16")) -> R.Tensor((32, 8, 16, 8), dtype="float16"):
            R.func_attr({"Composite": "cutlass.attention", "Primitive": 1, "WorkspaceSize": 65536})
            with R.dataflow():
                gv_2: R.Tensor((32, 8, 16, 8), dtype="float16") = R.nn.attention(q_1, k_1, v_1, scale=None, causal_mask=None, window_size=None)
                R.output(gv_2)
            return gv_2

        gv1: R.Tensor((32, 8, 16, 8), dtype="float16") = gv(q, k, v)
        return gv1