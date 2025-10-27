# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(data: R.Tensor((2, "n"), dtype="int8"), scale: R.Tensor(("n",), dtype="float32"), zp: R.Tensor(("n",), dtype="int8")) -> R.Tensor((2, "n"), dtype="float32"):
        n = T.int64()
        out: R.Tensor((2, n), dtype="float32") = R.dequantize(data, scale, zp, out_dtype="float32", axis=-1)
        return out