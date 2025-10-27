# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((3, 3), dtype="float32") = R.grad.start_checkpoint(x)
            lv1: R.Tensor((3, 3), dtype="float32") = R.multiply(lv, R.const(2, "float32"))
            lv2: R.Tensor((3, 3), dtype="float32") = R.multiply(lv1, R.const(2, "float32"))
            lv3: R.Tensor((3, 3), dtype="float32") = R.grad.end_checkpoint(lv2)
            lv4: R.Tensor((3, 3), dtype="float32") = R.multiply(x, lv3)
            lv5: R.Tensor((3, 3), dtype="float32") = R.grad.start_checkpoint(lv4)
            lv6: R.Tensor((3, 3), dtype="float32") = R.multiply(lv5, R.const(2, "float32"))
            lv7: R.Tensor((3, 3), dtype="float32") = R.multiply(lv6, R.const(2, "float32"))
            lv8: R.Tensor((3, 3), dtype="float32") = R.grad.end_checkpoint(lv7)
            lv9: R.Tensor((3, 3), dtype="float32") = R.multiply(lv4, lv8)
            lv10: R.Tensor((3, 3), dtype="float32") = R.grad.start_checkpoint(lv9)
            lv11: R.Tensor((3, 3), dtype="float32") = R.multiply(lv10, R.const(2, "float32"))
            lv12: R.Tensor((3, 3), dtype="float32") = R.multiply(lv11, R.const(2, "float32"))
            lv13: R.Tensor((3, 3), dtype="float32") = R.grad.end_checkpoint(lv12)
            lv14: R.Tensor((3, 3), dtype="float32") = R.multiply(lv9, lv13)
            gv: R.Tensor((), dtype="float32") = R.sum(lv14, axis=None, keepdims=False)
            R.output(gv)
        return gv