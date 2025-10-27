# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 10), dtype="float32"):
        gv0: R.Tensor((20, 10), dtype="float32") = R.ccl.allgather(x, R.prim_value(2))
        gv1: R.Tensor((20, 10), dtype="float32") = R.ccl.allgather(x, R.prim_value(2))
        return x