# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(s: R.Shape(["c"]), x: R.Tensor(("n", "4 * c", "h", "w"), dtype="float32"), gamma: R.Tensor(("4 * c",), dtype="float32"), beta: R.Tensor(("4 * c",), dtype="float32")) -> R.Tensor(("n", "4 * c", "h", "w"), dtype="float32"):
        n = T.int64()
        c = T.int64()
        h = T.int64()
        w = T.int64()
        gv: R.Tensor((n, 4 * c, h, w), dtype="float32") = R.nn.group_norm(x, gamma, beta, num_groups=4, channel_axis=1, axes=[2, 3], epsilon=1.0000000000000001e-05, center=True, scale=True)
        return gv