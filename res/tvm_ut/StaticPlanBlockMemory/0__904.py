# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def tir_exp(var_rxplaceholder: T.handle, var_compute: T.handle):
        T.evaluate(0)

    @R.function
    def main(x: R.Tensor(("n", "m", "T.max(n - m, 1)"), dtype="float32")) -> R.Tensor(("n", "m", "T.max(n - m, 1)"), dtype="float32"):
        n = T.int64()
        m = T.int64()
        R.func_attr({"relax.force_pure": 1, "tir_var_upper_bound": {"m": 5, "n": 20}})
        cls = Module
        alloc: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = R.builtin.alloc_tensor(R.shape([n, m, T.max(n - m, 1)]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.tir_exp(x, alloc)
        y: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = alloc
        alloc1: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = R.builtin.alloc_tensor(R.shape([n, m, T.max(n - m, 1)]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.tir_exp(y, alloc1)
        z: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = alloc1
        alloc2: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = R.builtin.alloc_tensor(R.shape([n, m, T.max(n - m, 1)]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.tir_exp(z, alloc2)
        r: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = alloc2
        return r