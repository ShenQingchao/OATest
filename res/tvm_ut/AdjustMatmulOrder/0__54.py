# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((16,)), A: R.Tensor((32, "lora_r")), B: R.Tensor(("lora_r", 16))) -> R.Tensor((32,)):
        lora_r = T.int64()
        weight: R.Tensor((32, 16)) = R.matmul(A, B, out_dtype="void")
        out: R.Tensor((32,)) = R.matmul(weight, x, out_dtype="void")
        return out