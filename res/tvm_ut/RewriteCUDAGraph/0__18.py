# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def func1() -> R.Tuple:
        R.func_attr({"relax.force_pure": 1})
        storage1: R.Object = R.memory.alloc_storage(R.shape([128]), R.prim_value(0), R.str("global"), R.dtype("float32"))
        storage2: R.Object = R.memory.alloc_storage(R.shape([256]), R.prim_value(0), R.str("global"), R.dtype("float32"))
        storage3: R.Object = R.memory.alloc_storage(R.shape([512]), R.prim_value(0), R.str("ipc_memory"), R.dtype("float32"))
        alloc1: R.Tensor((128,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        alloc2: R.Tensor((256,), dtype="float32") = R.memory.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc3: R.Tensor((512,), dtype="float32") = R.memory.alloc_tensor(storage3, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        R.call_packed("dummy", alloc1, alloc2, alloc3, sinfo_args=(R.Tuple,))
        return R.tuple()

    @R.function
    def func2() -> R.Tuple:
        R.func_attr({"relax.force_pure": 1})
        storage1: R.Object = R.memory.alloc_storage(R.shape([192]), R.prim_value(0), R.str("global"), R.dtype("float32"))
        storage2: R.Object = R.memory.alloc_storage(R.shape([64]), R.prim_value(0), R.str("global"), R.dtype("float32"))
        storage3: R.Object = R.memory.alloc_storage(R.shape([1024]), R.prim_value(0), R.str("ipc_memory"), R.dtype("float32"))
        storage4: R.Object = R.memory.alloc_storage(R.shape([512]), R.prim_value(0), R.str("global"), R.dtype("float32"))
        alloc1: R.Tensor((192,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([192]), R.dtype("float32"))
        alloc2: R.Tensor((64,), dtype="float32") = R.memory.alloc_tensor(storage2, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        alloc3: R.Tensor((1024,), dtype="float32") = R.memory.alloc_tensor(storage3, R.prim_value(0), R.shape([1024]), R.dtype("float32"))
        alloc4: R.Tensor((512,), dtype="float32") = R.memory.alloc_tensor(storage4, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        R.call_packed("dummy", alloc1, alloc2, alloc3, alloc4, sinfo_args=(R.Tuple,))
        return R.tuple()