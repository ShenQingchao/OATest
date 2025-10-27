# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(A: R.Tensor((16,), dtype="int32"), B: R.Tensor((16,), dtype="int32")) -> R.Tensor((16,), dtype="int32"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            offset: R.Tensor((16,), dtype="int32") = R.ones(R.shape([16]), dtype="int32")
            A_offset: R.Tensor((16,), dtype="int32") = R.add(A, offset)
            B_offset: R.Tensor((16,), dtype="int32") = R.add(B, offset)
            output: R.Tensor((16,), dtype="int32") = R.multiply(A_offset, B_offset)
            R.output(output)
        return output