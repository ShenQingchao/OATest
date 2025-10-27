# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def foo(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((64,), dtype="int32"):
        tmp_buf1: R.Tensor((64,), dtype="int32") = R.builtin.alloc_tensor(R.shape([64]), R.dtype("int32"), R.prim_value(0), R.str("global"))
        tmp_buf2: R.Tensor((64,), dtype="int32") = R.builtin.alloc_tensor(R.shape([64]), R.dtype("int32"), R.prim_value(0), R.str("global"))
        out: R.Tensor((64,), dtype="int32") = R.add(tmp_buf1, tmp_buf2)
        return out