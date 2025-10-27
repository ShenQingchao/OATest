# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def add1(A: T.Buffer((T.int64(2), T.int64(3)), "bool"), B: T.Buffer((T.int64(2), T.int64(3)), "bool"), C: T.Buffer((T.int64(2), T.int64(3)), "bool")):
        T.evaluate(0)

    @R.function
    def main(y: R.Tensor((2, 3), dtype="bool")) -> R.Tensor((2, 3), dtype="bool"):
        R.func_attr({"relax.force_pure": 1})
        cls = Module
        alloc: R.Tensor((2, 3), dtype="bool") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("bool"), R.prim_value(0), R.str("global"))
        cls.add1(y, y, alloc)
        gv1: R.Tensor((2, 3), dtype="bool") = alloc
        return y