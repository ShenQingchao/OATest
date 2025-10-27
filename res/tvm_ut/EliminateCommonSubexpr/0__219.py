# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32"), condition: R.Prim("bool")) -> R.Tensor((2, 3), dtype="float32"):
        if condition:
            A: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
            B_then: R.Tensor((2, 3), dtype="float32") = R.multiply(x, A)
            B: R.Tensor((2, 3), dtype="float32") = B_then
        else:
            A: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
            B_else: R.Tensor((2, 3), dtype="float32") = R.multiply(y, A)
            B: R.Tensor((2, 3), dtype="float32") = B_else
        return B