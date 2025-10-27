# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((10, 10), dtype="float32")) -> R.Tuple(R.Tensor((10, 10), dtype="float32"), R.Tensor((10, 10), dtype="float32")):
        with R.dataflow():
            gv0: R.Tensor((10, 10), dtype="float32") = R.clip(x, R.prim_value(0), R.prim_value(4))
            gv1: R.Tensor((10, 10), dtype="float32") = R.clip(x, R.prim_value(1), R.prim_value(3))
            R.output(gv0, gv1)
        return (gv0, gv1)