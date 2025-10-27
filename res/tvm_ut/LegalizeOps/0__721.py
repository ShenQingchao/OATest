# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(data: R.Tensor((2, 4), dtype="float16")) -> R.Tensor((2, 4), dtype="int8"):
        out: R.Tensor((2, 4), dtype="int8") = R.quantize(data, R.const(2, "float16"), R.const(1, "int8"), out_dtype="int8", axis=-1)
        return out