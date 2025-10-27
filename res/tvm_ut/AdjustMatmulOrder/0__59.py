# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("batch_size", 4096)), A: R.Tensor((4096, "lora_r")), B: R.Tensor(("lora_r", 4096))) -> R.Tensor(("batch_size", 4096)):
        batch_size = T.int64()
        lora_r = T.int64()
        R.func_attr({"tir_var_upper_bound": {"lora_r": 2048}})
        linear_weight: R.Tensor((4096, 4096)) = R.matmul(A, B, out_dtype="void")
        matmul_weight: R.Tensor((4096, 4096)) = R.permute_dims(linear_weight, axes=None)
        out: R.Tensor((batch_size, 4096)) = R.matmul(x, matmul_weight, out_dtype="void")
        return out