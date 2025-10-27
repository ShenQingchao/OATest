# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((64, 32, 16), dtype="float32"), y: R.Prim("float32")) -> R.Tensor((64, 32, 16), dtype="float32"):
        gv: R.Tensor((64, 32, 16), dtype="float32") = R.floor_divide(x, y)
        return gv