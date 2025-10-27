# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    I.module_attrs({"system_lib_prefix": "hello_"})
    @T.prim_func
    def tir_zeros(x: T.Buffer((2,), "float32")):
        x[0] = T.float32(0)

    @R.function(private=True)
    def main() -> R.Tensor((2,), dtype="float32"):
        cls = Module
        gv0 = R.call_tir(cls.tir_zeros, R.tuple(), out_sinfo=R.Tensor((2,), dtype="float32"))
        return gv0