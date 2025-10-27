# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(q: R.Tensor((4, 16, 32, 8), dtype="float32"), k: R.Tensor((4, 8, 32, 8), dtype="float32"), v: R.Tensor((4, 8, 32, 16), dtype="float32"), bias: R.Tensor((4, 32, 16, 8), dtype="float32")) -> R.Tensor((4, 16, 32, 16), dtype="float32"):
        gv: R.Tensor((4, 16, 32, 16), dtype="float32") = R.nn.attention(q, k, v, bias, scale=T.float32(0.10000000000000001), causal_mask="TopLeft", window_size=None)
        return gv
