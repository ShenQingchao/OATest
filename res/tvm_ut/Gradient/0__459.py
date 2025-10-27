# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            x_scp: R.Tensor((3, 3), dtype="float32") = R.grad.start_checkpoint(x)
            lv: R.Tensor((3, 3), dtype="float32") = R.multiply(x_scp, x_scp)
            lv1: R.Tensor((3, 3), dtype="float32") = R.multiply(lv, x_scp)
            lv1_ecp: R.Tensor((3, 3), dtype="float32") = R.grad.end_checkpoint(lv1)
            gv: R.Tensor((), dtype="float32") = R.sum(lv1_ecp, axis=None, keepdims=False)
            R.output(gv)
        return gv