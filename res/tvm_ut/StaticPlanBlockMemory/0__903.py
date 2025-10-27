# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def add(rxplaceholder: T.handle, rxplaceholder_1: T.handle, T_add: T.handle):
        T.evaluate(0)

    @T.prim_func
    def exp(rxplaceholder: T.handle, compute: T.handle):
        T.evaluate(0)

    @T.prim_func
    def log(rxplaceholder: T.handle, compute: T.handle):
        T.evaluate(0)

    @T.prim_func
    def pad(rxplaceholder: T.handle, PadInput: T.handle):
        T.evaluate(0)

    @T.prim_func
    def relu(rxplaceholder: T.handle, compute: T.handle):
        T.evaluate(0)

    @T.prim_func
    def reshape(rxplaceholder: T.handle, T_reshape: T.handle):
        T.evaluate(0)

    @R.function
    def main(x: R.Tensor((2, "n"), dtype="float32")) -> R.Tensor(("2 * n + 2",), dtype="float32"):
        n = T.int64()
        R.func_attr({"relax.force_pure": 1, "tir_var_upper_bound": {"n": 4}})
        cls = Module
        alloc: R.Tensor((2, n), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, n]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.exp(x, alloc)
        lv: R.Tensor((2, n), dtype="float32") = alloc
        lv1: R.Tensor((2 * n,), dtype="float32") = R.reshape(lv, R.shape([2 * n]))
        alloc1: R.Tensor((2 * n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([2 * n]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.relu(lv1, alloc1)
        lv2: R.Tensor((2 * n,), dtype="float32") = alloc1
        alloc2: R.Tensor((2 * n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([2 * n]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.add(lv2, R.const(1, "float32"), alloc2)
        lv3: R.Tensor((2 * n,), dtype="float32") = alloc2
        alloc3: R.Tensor((2 * n + 2,), dtype="float32") = R.builtin.alloc_tensor(R.shape([2 * n + 2]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.pad(lv3, alloc3)
        lv4: R.Tensor((2 * n + 2,), dtype="float32") = alloc3
        alloc4: R.Tensor((2 * n + 2,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.log(lv4, alloc4)
        gv: R.Tensor((2 * n + 2,), dtype="float32") = alloc4
        return gv