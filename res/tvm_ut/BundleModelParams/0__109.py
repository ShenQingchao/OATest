# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(a: R.Tensor((16,), dtype="float32"), b: R.Tensor((16,), dtype="float32"), c: R.Tensor((16,), dtype="float32")) -> R.Tensor((16,), dtype="float32"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            expr: R.Tensor((16,), dtype="float32") = a
            expr_1: R.Tensor((16,), dtype="float32") = R.add(expr, b)
            expr_2: R.Tensor((16,), dtype="float32") = R.add(expr_1, c)
            R.output(expr_2)
        return expr_2