# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("n", "h", "w", "c"), dtype="float32"), gamma: R.Tensor(("c",), dtype="float32"), beta: R.Tensor(("c",), dtype="float32"), moving_mean: R.Tensor(("c",), dtype="float32"), moving_var: R.Tensor(("c",), dtype="float32")) -> R.Tuple(R.Tensor(("n", "h", "w", "c"), dtype="float32"), R.Tensor(("c",), dtype="float32"), R.Tensor(("c",), dtype="float32")):
        n = T.int64()
        h = T.int64()
        w = T.int64()
        c = T.int64()
        gv: R.Tuple(R.Tensor((n, h, w, c), dtype="float32"), R.Tensor((c,), dtype="float32"), R.Tensor((c,), dtype="float32")) = R.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
        return gv