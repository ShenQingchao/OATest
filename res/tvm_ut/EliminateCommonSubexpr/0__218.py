# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32"), condition: R.Prim("bool")) -> R.Tensor((2, 3), dtype="float32"):
        A: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
        if condition:
            B: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
            C: R.Tensor((2, 3), dtype="float32") = R.multiply(x, B)
            D_then: R.Tensor((2, 3), dtype="float32") = R.multiply(A, C)
            D: R.Tensor((2, 3), dtype="float32") = D_then
        else:
            B: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
            C: R.Tensor((2, 3), dtype="float32") = R.multiply(y, B)
            D_else: R.Tensor((2, 3), dtype="float32") = R.multiply(A, C)
            D: R.Tensor((2, 3), dtype="float32") = D_else
        return D