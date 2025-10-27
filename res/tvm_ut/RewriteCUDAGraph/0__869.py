# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0_i1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for i0_i1_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                with T.block("compute"):
                    i0 = T.axis.spatial(T.int64(2), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) // T.int64(4))
                    i1 = T.axis.spatial(T.int64(4), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) % T.int64(4))
                    T.reads(rxplaceholder[i0, i1])
                    T.writes(compute[i0, i1])
                    compute[i0, i1] = T.exp(rxplaceholder[i0, i1])

    @R.function
    def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
        R.func_attr({"relax.force_pure": 1})
        cls = Module
        storage: R.Object = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
        alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
        cls.exp(x, alloc)
        storage1: R.Object = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
        alloc1: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
        cls.exp(alloc, alloc1)
        R.memory.kill_tensor(alloc)
        alloc2: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
        cls.exp(alloc1, alloc2)
        R.memory.kill_tensor(alloc1)
        storage2: R.Object = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
        alloc3: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage2, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
        cls.exp(alloc2, alloc3)
        R.memory.kill_tensor(alloc2)
        alloc4: R.Tensor((2, 4), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 4]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.exp(alloc3, alloc4)
        R.memory.kill_tensor(alloc3)
        R.memory.kill_storage(storage)
        R.memory.kill_storage(storage1)
        R.memory.kill_storage(storage2)
        return alloc4