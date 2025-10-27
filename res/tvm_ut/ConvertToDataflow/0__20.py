# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
        with R.dataflow():
            z: R.Tensor = R.add(x, y)
            w: R.Tensor = R.multiply(z, y)
            R.output(w)
        v: R.Tensor = R.add(w, x)
        return v