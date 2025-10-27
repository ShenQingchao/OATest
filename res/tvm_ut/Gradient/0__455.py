# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            x_scp: R.Tensor((3, 3), dtype="float32") = R.grad.start_checkpoint(x)
            lv1: R.Tensor((3, 3), dtype="float32") = R.power(x_scp, R.const(3, "float32"))
            lv2: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = x, lv1
            lv3: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = lv2
            lv4: R.Tensor((3, 3), dtype="float32") = lv3[0]
            lv4_1: R.Tensor((3, 3), dtype="float32") = R.power(lv4, R.const(3, "float32"))
            lv4_ecp: R.Tensor((3, 3), dtype="float32") = R.grad.end_checkpoint(lv4_1)
            gv: R.Tensor((), dtype="float32") = R.sum(lv4_ecp, axis=None, keepdims=False)
            R.output(gv)
        return gv