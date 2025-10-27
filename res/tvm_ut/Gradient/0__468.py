# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def f_mul(var_a: T.handle, var_b: T.handle, var_f_mul: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        a = T.match_buffer(var_a, (n, n))
        b = T.match_buffer(var_b, (n, n))
        f_mul = T.match_buffer(var_f_mul, (n, n))
        # with T.block("root"):
        for i0, i1 in T.grid(n, n):
            with T.block("f_mul"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(a[v_i0, v_i1], b[v_i0, v_i1])
                T.writes(f_mul[v_i0, v_i1])
                f_mul[v_i0, v_i1] = a[v_i0, v_i1] * b[v_i0, v_i1]

    @R.function
    def main(a: R.Tensor(("n", "n"), dtype="float32"), b: R.Tensor(("n", "n"), dtype="float32")) -> R.Tensor((), dtype="float32"):
        n = T.int64()
        cls = Module
        with R.dataflow():
            lv: R.Tensor((n, n), dtype="float32") = R.call_tir_with_grad(cls.f_mul, (a, b), out_sinfo=R.Tensor((n, n), dtype="float32"), te_grad_name="f_mul_grad")
            gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
            R.output(gv)
        return gv