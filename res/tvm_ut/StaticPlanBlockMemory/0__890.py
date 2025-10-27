# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def add(A: T.Buffer((T.int64(2), T.int64(3)), "float32"), B: T.Buffer((T.int64(2), T.int64(3)), "float32"), C: T.Buffer((T.int64(2), T.int64(3)), "float32")):
        T.evaluate(0)

    @T.prim_func
    def add1(A: T.Buffer((T.int64(2), T.int64(3)), "int32"), B: T.Buffer((T.int64(2), T.int64(3)), "int32"), C: T.Buffer((T.int64(2), T.int64(3)), "int32")):
        T.evaluate(0)

    @R.function
    def main(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="int32")) -> R.Tensor((2, 3), dtype="float32"):
        R.func_attr({"relax.force_pure": 1})
        cls = Module
        alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.add(x, x, alloc)
        gv: R.Tensor((2, 3), dtype="float32") = alloc
        alloc1: R.Tensor((2, 3), dtype="int32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("int32"), R.prim_value(0), R.str("global"))
        cls.add1(y, y, alloc1)
        gv1: R.Tensor((2, 3), dtype="int32") = alloc1
        return x