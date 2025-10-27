# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("m", "n"), dtype="float32"), weight: R.Tensor(("m * n",), dtype="float32")) -> R.Tensor(("m", "n"), dtype="float32"):
        m = T.int64()
        n = T.int64()
        R.func_attr({"num_input": 1})
        with R.dataflow():
            weight_1: R.Tensor((m * n,), dtype="float32") = R.add(weight, R.const(1, "float32"))
            weight_2: R.Tensor((m, n), dtype="float32") = R.reshape(weight_1, R.shape([m, n]))
            output: R.Tensor((m, n), dtype="float32") = R.multiply(x, weight_2)
            R.output(output)
        return output