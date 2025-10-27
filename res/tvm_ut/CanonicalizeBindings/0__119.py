# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(w: R.Tensor) -> R.Tuple(R.Tensor, R.Tensor):
        with R.dataflow():
            x: R.Tensor = R.add(w, R.const(1, "int32"))
            y: R.Tensor = x
            z: R.Tensor = x
            R.output(y, z)
        return (y, z)