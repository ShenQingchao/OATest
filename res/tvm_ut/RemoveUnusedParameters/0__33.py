# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def func(A: R.Tensor, B: R.Tensor) -> R.Tensor:
        return A

    @R.function
    def main(A: R.Tensor, B: R.Tensor) -> R.Tensor:
        cls = Module
        gv: R.Tensor = cls.func(A, B)
        return gv