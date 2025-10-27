# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def all_less_than_zero(A: T.Buffer((2, 3), "float32"), B: T.Buffer((), "bool")):
        T.evaluate(0)

    @T.prim_func
    def exp(A: T.Buffer((2, 3), "float32"), B: T.Buffer((2, 3), "float32")):
        T.evaluate(0)

    @R.function
    def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
        R.func_attr({"relax.force_pure": 1})
        cls = Module
        alloc: R.Tensor((), dtype="bool") = R.builtin.alloc_tensor(R.shape([]), R.dtype("bool"), R.prim_value(0), R.str("global"))
        cls.all_less_than_zero(x, alloc)
        x1: R.Tensor((), dtype="bool") = alloc
        if x1:
            y: R.Tensor((2, 3), dtype="float32") = x
        else:
            alloc1: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
            cls.exp(x, alloc1)
            gv3: R.Tensor((2, 3), dtype="float32") = alloc1
            y: R.Tensor((2, 3), dtype="float32") = gv3
        return x