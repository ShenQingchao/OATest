# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(q: R.Tensor((4, "seq_len", 32, 8), dtype="float32"), k: R.Tensor((4, "seq_len_kv", 32, 8), dtype="float32"), v: R.Tensor((4, "seq_len_kv", 32, 16), dtype="float32"), bias: R.Tensor((4, 32, "seq_len", "seq_len_kv"), dtype="float32")) -> R.Tensor((4, "seq_len", 32, 16), dtype="float32"):
        seq_len = T.int64()
        seq_len_kv = T.int64()
        gv: R.Tensor((4, seq_len, 32, 16), dtype="float32") = R.nn.attention(q, k, v, bias, scale=T.float32(0.10000000000000001), causal_mask="BottomRight", window_size=None)
        return gv
