# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor) -> R.Tensor:
        with R.dataflow():
            y: R.Tensor = R.add(x, R.const(1, "int32"))
            z: R.Tuple(R.Tensor) = (y,)
            lv2: R.Tensor = z[0]
            gv: R.Tensor = R.add(lv2, R.const(1, "int32"))
            R.output(z, gv)
        return gv