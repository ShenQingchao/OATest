# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(data: R.Tensor((4, "n"), dtype="float32"), scale: R.Tensor(("n",), dtype="float32"), zp: R.Tensor(("n",), dtype="int8")) -> R.Tensor((4, "n"), dtype="int8"):
        n = T.int64()
        out: R.Tensor((4, n), dtype="int8") = R.quantize(data, scale, zp, out_dtype="int8", axis=-1)
        return out