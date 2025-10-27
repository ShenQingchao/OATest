# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def func(args: R.Tuple(R.Tuple(R.Tensor, R.Tensor), R.Tuple(R.Tensor, R.Tensor))) -> R.Tensor:
        gv: R.Tuple(R.Tensor, R.Tensor) = args[0]
        gv1: R.Tensor = gv[1]
        return gv1

    @R.function
    def main(A: R.Tensor, B: R.Tensor, C: R.Tensor, D: R.Tensor) -> R.Tensor:
        cls = Module
        gv: R.Tensor = cls.func(((A, B), (C, D)))
        return gv