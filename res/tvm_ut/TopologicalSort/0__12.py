# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(A: R.Tensor) -> R.Tuple(R.Tensor, R.Tensor):
        with R.dataflow():
            B2: R.Tensor = R.add(A, R.const(2, "int32"))
            B1: R.Tensor = R.add(A, R.const(1, "int32"))
            C2: R.Tensor = R.add(A, B2)
            C1: R.Tensor = R.add(A, B1)
            R.output(C2, C1)
        return (C1, C2)