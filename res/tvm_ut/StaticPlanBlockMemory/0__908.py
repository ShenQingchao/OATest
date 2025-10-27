# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def exp(A: T.handle, B: T.handle):
        T.evaluate(0)

    @R.function
    def func1(x: R.Tensor((8,), dtype="float32")) -> R.Tensor((8,), dtype="float32"):
        R.func_attr({"relax.force_pure": 1})
        cls = Module
        alloc: R.Tensor((8,), dtype="float32") = R.builtin.alloc_tensor(R.shape([8]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.exp(x, alloc)
        lv: R.Tensor((8,), dtype="float32") = alloc
        alloc1: R.Tensor((8,), dtype="float32") = R.builtin.alloc_tensor(R.shape([8]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.exp(lv, alloc1)
        gv: R.Tensor((8,), dtype="float32") = alloc1
        return gv

    @R.function
    def func2(x: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"relax.force_pure": 1})
        cls = Module
        alloc: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.exp(x, alloc)
        lv: R.Tensor((10,), dtype="float32") = alloc
        alloc1: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.exp(lv, alloc1)
        gv: R.Tensor((10,), dtype="float32") = alloc1
        return gv