# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def add_one(x_handle: T.handle, y_handle: T.handle):
        m = T.int64()
        x = T.match_buffer(x_handle, (m,))
        y = T.match_buffer(y_handle, (m,))
        # with T.block("root"):
        for i in range(m):
            with T.block("add"):
                vi = T.axis.spatial(m, i)
                T.reads(x[vi])
                T.writes(y[vi])
                y[vi] = x[vi] + T.float32(1)

    @R.function
    def main(x: R.Tensor(("m",), dtype="float32")) -> R.Tensor(("m",), dtype="float32"):
        m = T.int64()
        R.func_attr({"relax.force_pure": 1, "relax.rewrite_cuda_graph.capture_symbolic_vars": ["m"]})
        cls = Module
        storage: R.Object = R.memory.alloc_storage(R.shape([16]), R.prim_value(0), R.str("global"), R.dtype("float32"))
        alloc1: R.Tensor((m,), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([m]), R.dtype("float32"))
        cls.add_one(x, alloc1)
        storage1: R.Object = R.memory.alloc_storage(R.shape([16]), R.prim_value(0), R.str("global"), R.dtype("float32"))
        alloc2: R.Tensor((m,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([m]), R.dtype("float32"))
        cls.add_one(alloc1, alloc2)
        alloc3: R.Tensor((m,), dtype="float32") = R.builtin.alloc_tensor(R.shape([m]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.add_one(alloc2, alloc3)
        return alloc3