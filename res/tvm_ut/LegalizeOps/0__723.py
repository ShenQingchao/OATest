# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(data: R.Tensor((2, 4), dtype="int8")) -> R.Tensor((2, 4), dtype="float32"):
        out: R.Tensor((2, 4), dtype="float32") = R.dequantize(data, R.const(2, "float32"), R.const(1, "int8"), out_dtype="float32", axis=0)
        return out