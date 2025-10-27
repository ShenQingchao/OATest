# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main() -> R.Tuple:
        storage: R.Object = R.memory.alloc_storage(R.shape([8]), R.prim_value(0), R.str("global"), R.dtype("float32"))
        alloc3: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([8]), R.dtype("float32"))
        return R.tuple()