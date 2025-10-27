# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def func(args: R.Tuple(R.Tensor, R.Tensor)) -> R.Tensor:
        gv: R.Tensor = args[0]
        return gv

    @R.function
    def main(A: R.Tensor, B: R.Tensor) -> R.Tensor:
        cls = Module
        gv: R.Tensor = cls.func((A, B))
        return gv