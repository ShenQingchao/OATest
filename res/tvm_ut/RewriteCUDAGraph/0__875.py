# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(pure=False)
    def main() -> R.Tuple:
        storage0: R.Object = R.memory.alloc_storage(R.shape([8]), R.prim_value(0), R.str("global"), R.dtype("float32"))
        alloc0: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(storage0, R.prim_value(0), R.shape([8]), R.dtype("float32"))
        _: R.Object = R.call_packed("dummy_func", alloc0, R.dtype("float32"), R.str("string"))
        return R.tuple()