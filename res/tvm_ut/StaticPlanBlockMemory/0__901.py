# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def add(A: T.Buffer((T.int64(2), T.int64(25), T.int64(2)), "float32"), B: T.Buffer((T.int64(2), T.int64(25), T.int64(2)), "float32"), C: T.Buffer((T.int64(2), T.int64(25), T.int64(2)), "float32")):
        T.evaluate(0)

    @R.function
    def main(x: R.Tensor((2, 50), dtype="float32"), y: R.Tensor((100,), dtype="float32")) -> R.Tensor((2, 25, 2), dtype="float32"):
        R.func_attr({"relax.force_pure": 1})
        cls = Module
        lv: R.Tensor((2, 25, 2), dtype="float32") = R.reshape(x, R.shape([2, 25, 2]))
        lv1: R.Tensor((2, 25, 2), dtype="float32") = R.reshape(y, R.shape([2, 25, 2]))
        alloc: R.Tensor((2, 25, 2), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 25, 2]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.add(lv, lv1, alloc)
        gv: R.Tensor((2, 25, 2), dtype="float32") = alloc
        return gv