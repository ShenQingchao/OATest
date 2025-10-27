# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
        with R.dataflow():
            z: R.Tensor = R.add(x, y)
            R.output(z)
        if R.const(True, "bool"):
            with R.dataflow():
                w: R.Tensor = R.add(z, z)
                v: R.Tensor = R.multiply(w, w)
                R.output(w, v)
            q: R.Tensor = v
        else:
            with R.dataflow():
                w: R.Tensor = R.multiply(z, z)
                v: R.Tensor = R.add(w, w)
                R.output(w, v)
            q: R.Tensor = v
        return q