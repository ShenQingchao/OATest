# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((10, "n"), dtype="float32")) -> R.Tensor((3, "n"), dtype="float32"):
        n = T.int64()
        gv: R.Tensor((3, n), dtype="float32") = R.strided_slice(x, (R.prim_value(0),), (R.prim_value(1),), (R.prim_value(8),), (R.prim_value(3),), assume_inbound=False)
        return gv