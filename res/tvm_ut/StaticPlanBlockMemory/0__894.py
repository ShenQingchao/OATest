# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def exp(A: T.Buffer((2, 3), "float32"), B: T.Buffer((2, 3), "float32")):
        T.evaluate(0)

    @R.function
    def main(cond: R.Tensor((), dtype="bool"), x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
        R.func_attr({"relax.force_pure": 1})
        cls = Module
        alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.exp(x, alloc)
        y: R.Tensor((2, 3), dtype="float32") = alloc
        if cond:
            z: R.Tensor((2, 3), dtype="float32") = y
        else:
            z: R.Tensor((2, 3), dtype="float32") = y
        return x