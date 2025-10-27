# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def exp(var_A: T.handle, var_B: T.handle):
        m, n = T.int64(), T.int64()
        A = T.match_buffer(var_A, (m, n))
        B = T.match_buffer(var_B, (m, n))
        T.evaluate(0)

    @R.function
    def main(x: R.Tensor(("m", "n"), dtype="float32")) -> R.Tensor(("m", "n"), dtype="float32"):
        m = T.int64()
        n = T.int64()
        R.func_attr({"relax.force_pure": 1})
        cls = Module
        alloc: R.Tensor((m, n), dtype="float32") = R.builtin.alloc_tensor(R.shape([m, n]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.exp(x, alloc)
        y: R.Tensor((m, n), dtype="float32") = alloc
        return x