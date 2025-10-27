# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((8, 9, 10, 10), dtype="float32")) -> R.Tensor((4, 9, 10, 3), dtype="float32"):
        gv: R.Tensor((4, 9, 10, 3), dtype="float32") = R.strided_slice(x, (R.prim_value(0), R.prim_value(1), R.prim_value(3)), (R.prim_value(1), R.prim_value(0), R.prim_value(8)), (R.prim_value(8), R.prim_value(9), R.prim_value(0)), (R.prim_value(2), R.prim_value(1), R.prim_value(-3)), assume_inbound=False)
        return gv