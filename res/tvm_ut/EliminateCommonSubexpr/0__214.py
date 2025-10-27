# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(pure=False)
    def foo(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3)):
        y: R.Tensor((2, 3)) = R.call_packed("extern_func_name", x, sinfo_args=(R.Tensor((2, 3)),))
        z: R.Tensor((2, 3)) = R.call_packed("extern_func_name", y, sinfo_args=(R.Tensor((2, 3)),))
        return z