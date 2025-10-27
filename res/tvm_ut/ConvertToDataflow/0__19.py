# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
        z: R.Tensor = R.add(x, y)
        with R.dataflow():
            w: R.Tensor = R.multiply(z, y)
            v: R.Tensor = R.add(w, x)
            R.output(v)
        return v