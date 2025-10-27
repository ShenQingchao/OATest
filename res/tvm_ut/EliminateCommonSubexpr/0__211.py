# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")) -> R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")):
        with R.dataflow():
            A: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
            B: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
            C: R.Tensor((2, 3), dtype="float32") = R.multiply(A, B)
            R.output(B, C)
        D: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
        return (B, C, D)