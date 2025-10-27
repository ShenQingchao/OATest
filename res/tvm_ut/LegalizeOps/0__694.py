# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3,), dtype="float32"), y: R.Tensor((3,), dtype="float32")) -> R.Tensor(dtype="float32", ndim=2):
        gv: R.Tensor((), dtype="float32") = R.nn.cross_entropy_with_logits(x, y)
        return gv