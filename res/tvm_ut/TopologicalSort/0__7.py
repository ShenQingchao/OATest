# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(A: R.Tensor, B1: R.Tensor, B2: R.Tensor) -> R.Tensor:
        with R.dataflow():
            C2: R.Tensor = R.add(A, B2)
            C1: R.Tensor = R.add(A, B1)
            D2: R.Tensor = R.add(A, C2)
            D1: R.Tensor = R.add(A, C1)
            E: R.Tensor = R.add(D1, D2)
            R.output(E)
        return E