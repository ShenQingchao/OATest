# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((16,)), A: R.Tensor((16, 2)), B: R.Tensor((2, 32))) -> R.Tensor((32,)):
        R.func_attr({"num_input": 1})
        weight: R.Tensor((16, 32)) = R.matmul(A, B, out_dtype="void")
        out: R.Tensor((32,)) = R.matmul(x, weight, out_dtype="void")
        return out