# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("n", "s", "f"), dtype="float32"), weight: R.Tensor(("s", "f"), dtype="float32")) -> R.Tensor(("n", "s", "f"), dtype="float32"):
        n = T.int64()
        s = T.int64()
        f = T.int64()
        gv: R.Tensor((n, s, f), dtype="float32") = R.nn.rms_norm(x, weight, axes=[1, 2], epsilon=1.0000000000000001e-05)
        return gv