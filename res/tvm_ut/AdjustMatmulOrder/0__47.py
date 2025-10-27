# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((16,)), A: R.Tensor((16, "lora_r")), B: R.Tensor(("lora_r", 32))) -> R.Tensor((32,)):
        lora_r = T.int64()
        weight: R.Tensor((16, 32)) = R.matmul(A, B, out_dtype="void")
        out: R.Tensor((32,)) = R.matmul(x, weight, out_dtype="void")
        return out