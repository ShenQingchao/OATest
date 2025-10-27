# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            x_scp: R.Tensor((3, 3), dtype="float32") = R.grad.start_checkpoint(x)
            lv1: R.Tensor((3, 3), dtype="float32") = R.power(x_scp, R.const(3, "float32"))
            lv1_ecp: R.Tensor((3, 3), dtype="float32") = R.grad.end_checkpoint(lv1)
            lv2: R.Tensor((3, 3), dtype="float32") = R.power(lv1_ecp, R.const(3, "float32"))
            lv2_scp: R.Tensor((3, 3), dtype="float32") = R.grad.start_checkpoint(lv2)
            lv3: R.Tensor((3, 3), dtype="float32") = R.power(lv2_scp, R.const(3, "float32"))
            lv4: R.Tensor((3, 3), dtype="float32") = R.power(lv3, R.const(3, "float32"))
            gv: R.Tensor((), dtype="float32") = R.sum(lv4, axis=None, keepdims=False)
            gv_ecp: R.Tensor((), dtype="float32") = R.grad.end_checkpoint(gv)
            R.output(gv_ecp)
        return gv_ecp