# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def tir_exp(var_rxplaceholder: T.handle, var_compute: T.handle):
        T.evaluate(0)

    @T.prim_func
    def tir_full(var_full: T.handle, n: T.int64):
        T.evaluate(0)

    @R.function
    def main(s: R.Shape(["n"])) -> R.Tensor(("n",), dtype="float32"):
        n = T.int64()
        R.func_attr({"relax.force_pure": 1, "tir_var_upper_bound": {"n": 20}})
        cls = Module
        alloc: R.Tensor((n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([n]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        R.vm.call_tir_dyn(cls.tir_full, (alloc, R.shape([n])))
        full: R.Tensor((n,), dtype="float32") = alloc
        alloc1: R.Tensor((n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([n]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.tir_exp(full, alloc1)
        lv2: R.Tensor((n,), dtype="float32") = alloc1
        alloc2: R.Tensor((n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([n]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.tir_exp(lv2, alloc2)
        lv3: R.Tensor((n,), dtype="float32") = alloc2
        return lv3