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
        lv1: R.Tensor((8,), dtype="float32") = R.reshape(alloc, R.shape([8]))
        R.memory.kill_tensor(alloc)
        storage1: R.Object = R.memory.alloc_storage(R.shape([40]), R.prim_value(0), R.str("global"), R.dtype("float32"))
        alloc1: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([8]), R.dtype("float32"))
        cls.relu(lv1, alloc1)
        R.memory.kill_tensor(lv1)
        alloc2: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([8]), R.dtype("float32"))
        R.memory.kill_storage(storage)
        cls.add(alloc1, R.const(1, "float32"), alloc2)
        R.memory.kill_tensor(alloc1)
        alloc3: R.Tensor((10,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([10]), R.dtype("float32"))
        R.memory.kill_storage(storage1)
        cls.pad(alloc2, alloc3)
        R.memory.kill_tensor(alloc2)
        storage_1: R.Object = R.memory.alloc_storage(R.shape([40]), R.prim_value(0), R.str("global"), R.dtype("uint8"))
        alloc4: R.Tensor((10,), dtype="float32") = R.memory.alloc_tensor(storage_1, R.prim_value(0), R.shape([10]), R.dtype("float32"))
        R.memory.kill_storage(storage_1)
        cls.log(alloc3, alloc4)
        R.memory.kill_tensor(alloc3)
        return alloc4