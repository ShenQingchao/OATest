# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32"), z: R.Tensor((3, 3), dtype="float32"), u: R.Tensor((3, 3), dtype="float32"), v: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            lv1: R.Tensor((3, 3), dtype="float32") = R.multiply(x, y)
            lv1_scp: R.Tensor((3, 3), dtype="float32") = R.grad.start_checkpoint(lv1)
            z_scp: R.Tensor((3, 3), dtype="float32") = R.grad.start_checkpoint(z)
            lv2: R.Tensor((3, 3), dtype="float32") = R.multiply(lv1_scp, z_scp)
            lv2_ecp: R.Tensor((3, 3), dtype="float32") = R.grad.end_checkpoint(lv2)
            u_scp: R.Tensor((3, 3), dtype="float32") = R.grad.start_checkpoint(u)
            v_scp: R.Tensor((3, 3), dtype="float32") = R.grad.start_checkpoint(v)
            lv3: R.Tensor((3, 3), dtype="float32") = R.multiply(u_scp, v_scp)
            lv3_ecp: R.Tensor((3, 3), dtype="float32") = R.grad.end_checkpoint(lv3)
            lv4: R.Tensor((3, 3), dtype="float32") = R.multiply(lv2_ecp, lv3_ecp)
            gv: R.Tensor((), dtype="float32") = R.sum(lv4, axis=None, keepdims=False)
            R.output(gv)
        return gv