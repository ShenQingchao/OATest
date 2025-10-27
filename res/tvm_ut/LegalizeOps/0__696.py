# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("n", "m"), dtype="float32"), y: R.Tensor(("n", "m"), dtype="float32")) -> R.Tensor(dtype="float32", ndim=2):
        n = T.int64()
        m = T.int64()
        gv: R.Tensor((), dtype="float32") = R.nn.cross_entropy_with_logits(x, y)
        return gv