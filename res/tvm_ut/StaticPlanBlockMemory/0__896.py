# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def exp(A: T.Buffer((2, 3), "float32"), B: T.Buffer((2, 3), "float32")):
        T.evaluate(0)

    @R.function
    def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
        R.func_attr({"relax.force_pure": 1})
        cls = Module
        alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.exp(x, alloc)
        y1: R.Tensor((2, 3), dtype="float32") = alloc
        alloc1: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.exp(x, alloc1)
        y2: R.Tensor((2, 3), dtype="float32") = alloc1
        alloc2: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.exp(x, alloc2)
        y3: R.Tensor((2, 3), dtype="float32") = alloc2
        t: R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")) = y1, y2
        nt: R.Tuple(R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")), R.Tensor((2, 3), dtype="float32")) = t, y3
        nt0: R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")) = nt[0]
        y1_: R.Tensor((2, 3), dtype="float32") = nt0[0]
        y2_: R.Tensor((2, 3), dtype="float32") = nt0[1]
        y3_: R.Tensor((2, 3), dtype="float32") = nt[1]
        alloc3: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.exp(y1_, alloc3)
        z1: R.Tensor((2, 3), dtype="float32") = alloc3
        alloc4: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.exp(y2_, alloc4)
        z2: R.Tensor((2, 3), dtype="float32") = alloc4
        alloc5: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.exp(y3_, alloc5)
        z3: R.Tensor((2, 3), dtype="float32") = alloc5
        return x