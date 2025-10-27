# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(data: R.Tensor((2, 4), dtype="float16"), scale: R.Tensor((2,), dtype="float16"), zp: R.Tensor((2,), dtype="int8")) -> R.Tensor((2, 4), dtype="uint8"):
        out: R.Tensor((2, 4), dtype="uint8") = R.quantize(data, scale, zp, out_dtype="uint8", axis=0)
        return out