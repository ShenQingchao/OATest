# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((16,)), A: R.Tensor((16, "lora_r")), B: R.Tensor(("lora_r", 32))) -> R.Tensor((32,)):
        lora_r = T.int64()
        x_1: R.Tensor((lora_r,)) = R.matmul(x, A, out_dtype="void")
        x_2: R.Tensor((32,)) = R.matmul(x_1, B, out_dtype="void")
        return x_2