# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(pure=False)
    def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
        alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        R.call_packed("extern_func", x, alloc, sinfo_args=(R.Tuple,))
        y: R.Tensor((2, 3), dtype="float32") = alloc
        alloc1: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        R.call_packed("extern_func", y, alloc1, sinfo_args=(R.Tuple,))
        z: R.Tensor((2, 3), dtype="float32") = alloc1
        return z