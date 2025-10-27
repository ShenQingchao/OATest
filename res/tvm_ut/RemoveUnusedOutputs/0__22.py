# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def func() -> R.Tuple(R.Tensor((16, 16), dtype="int32"), R.Tensor((16, 16), dtype="int32"), R.Tensor((32, 32), dtype="int32")):
        A: R.Tensor((16, 16), dtype="int32") = R.zeros(R.shape([16, 16]), dtype="int32")
        B: R.Tensor((16, 16), dtype="int32") = R.ones(R.shape([16, 16]), dtype="int32")
        C: R.Tensor((32, 32), dtype="int32") = R.zeros(R.shape([32, 32]), dtype="int32")
        return (A, B, C)

    @R.function
    def main_a() -> R.Tensor:
        cls = Module
        args: R.Tuple(R.Tensor, R.Tensor, R.Tensor) = cls.func()
        gv1: R.Tensor = args[0]
        return gv1

    @R.function
    def main_b() -> R.Tensor:
        cls = Module
        args: R.Tuple(R.Tensor, R.Tensor, R.Tensor) = cls.func()
        gv1: R.Tensor = args[2]
        return gv1