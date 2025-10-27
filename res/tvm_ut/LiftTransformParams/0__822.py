# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(shape: R.Shape(["n"])) -> R.Shape(["n"]):
        n = T.int64()
        R.func_attr({"num_input": 1})
        with R.dataflow():
            zeros: R.Tensor((n, n), dtype="float32") = R.zeros(R.shape([n, n]), dtype="float32")
            R.output()
        return shape