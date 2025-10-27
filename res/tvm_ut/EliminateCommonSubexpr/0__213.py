# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((6,), dtype="float32"):
        x_1: R.Tensor((6,), dtype="float32") = R.reshape(x, R.shape([6]))
        y_1: R.Tensor((6,), dtype="float32") = R.reshape(y, R.shape([6]))
        z: R.Tensor((6,), dtype="float32") = R.add(x_1, y_1)
        return z