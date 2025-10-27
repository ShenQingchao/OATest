# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def add(rxplaceholder: T.Buffer((T.int64(8),), "float32"), rxplaceholder_1: T.Buffer((), "float32"), T_add: T.Buffer((T.int64(8),), "float32")):
        T.evaluate(0)

    @T.prim_func
    def exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4)), "float32")):
        T.evaluate(0)

    @T.prim_func
    def log(rxplaceholder: T.Buffer((T.int64(10),), "float32"), compute: T.Buffer((T.int64(10),), "float32")):
        T.evaluate(0)

    @T.prim_func
    def pad(rxplaceholder: T.Buffer((T.int64(8),), "float32"), PadInput: T.Buffer((T.int64(10),), "float32")):
        T.evaluate(0)

    @T.prim_func
    def relu(rxplaceholder: T.Buffer((T.int64(8),), "float32"), compute: T.Buffer((T.int64(8),), "float32")):
        T.evaluate(0)

    @T.prim_func
    def reshape(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), T_reshape: T.Buffer((T.int64(8),), "float32")):
        T.evaluate(0)

    @R.function
    def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"relax.force_pure": 1})
        cls = Module
        storage: R.Object = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
        alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
        cls.exp(x, alloc)
        lv: R.Tensor((2, 4), dtype="float32") = alloc
        lv1: R.Tensor((8,), dtype="float32") = R.reshape(lv, R.shape([8]))
        storage1: R.Object = R.memory.alloc_storage(R.shape([40]), R.prim_value(0), R.str("global"), R.dtype("float32"))
        alloc1: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([8]), R.dtype("float32"))
        cls.relu(lv1, alloc1)
        lv2: R.Tensor((8,), dtype="float32") = alloc1
        alloc2: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([8]), R.dtype("float32"))
        cls.add(lv2, R.const(1, "float32"), alloc2)
        lv3: R.Tensor((8,), dtype="float32") = alloc2
        alloc3: R.Tensor((10,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([10]), R.dtype("float32"))
        cls.pad(lv3, alloc3)
        lv4: R.Tensor((10,), dtype="float32") = alloc3
        alloc4: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.log(lv4, alloc4)
        gv: R.Tensor((10,), dtype="float32") = alloc4
        return gv