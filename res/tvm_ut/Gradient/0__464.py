# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def f_mul(a: T.Buffer((T.int64(5), T.int64(5)), "float32"), b: T.Buffer((T.int64(5), T.int64(5)), "float32"), f_mul: T.Buffer((T.int64(5), T.int64(5)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(5), T.int64(5)):
            with T.block("f_mul"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(a[v_i0, v_i1], b[v_i0, v_i1])
                T.writes(f_mul[v_i0, v_i1])
                f_mul[v_i0, v_i1] = a[v_i0, v_i1] * b[v_i0, v_i1]

    @R.function
    def main(a: R.Tensor((5, 5), dtype="float32"), b: R.Tensor((5, 5), dtype="float32")) -> R.Tensor((), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((5, 5), dtype="float32") = R.call_tir_with_grad(cls.f_mul, (a, b), out_sinfo=R.Tensor((5, 5), dtype="float32"), te_grad_name="f_mul_grad")
            gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
            R.output(gv)
        return gv