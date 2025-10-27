# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((8,), dtype="float16"), w: R.Tensor(("m",))) -> R.Tuple(R.Tensor((8,), dtype="float16")):
        m = T.int64()
        R.func_attr({"num_input": 1, "relax.force_pure": 1})
        storage1: R.Object = R.memory.alloc_storage(R.shape([8]), R.prim_value(0), R.str("global"), R.dtype("float16"))
        alloc1: R.Tensor((8,), dtype="float16") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([8]), R.dtype("float16"))
        R.call_packed("dummy", x, w, alloc1, sinfo_args=(R.Tuple,))
        storage2: R.Object = R.memory.alloc_storage(R.shape([8]), R.prim_value(0), R.str("global"), R.dtype("float16"))
        alloc2: R.Tensor((8,), dtype="float16") = R.memory.alloc_tensor(storage2, R.prim_value(0), R.shape([8]), R.dtype("float16"))
        R.call_packed("dummy", alloc1, w, alloc2, sinfo_args=(R.Tuple,))
        storage3: R.Object = R.memory.alloc_storage(R.shape([8]), R.prim_value(0), R.str("global"), R.dtype("float16"))
        alloc3: R.Tensor((8,), dtype="float16") = R.memory.alloc_tensor(storage3, R.prim_value(0), R.shape([8]), R.dtype("float16"))
        R.call_packed("dummy", alloc2, w, alloc3, sinfo_args=(R.Tuple,))
        gv: R.Tuple(R.Tensor((8,), dtype="float16")) = (alloc3,)
        return gv