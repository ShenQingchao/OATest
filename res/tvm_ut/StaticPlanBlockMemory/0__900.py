# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
        R.func_attr({"relax.force_pure": 1})
        alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        return x