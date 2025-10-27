# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 10), dtype="float32"):
        gv0: R.Tensor((10, 10), dtype="float32") = R.ccl.allreduce(x, op_type="sum")
        gv1: R.Tensor((10, 10), dtype="float32") = R.ccl.allreduce(x, op_type="prod")
        gv2: R.Tensor((10, 10), dtype="float32") = R.ccl.allreduce(x, op_type="min")
        gv3: R.Tensor((10, 10), dtype="float32") = R.ccl.allreduce(x, op_type="max")
        gv4: R.Tensor((10, 10), dtype="float32") = R.ccl.allreduce(x, op_type="avg")
        return x