# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("a", "b"), dtype="float32")) -> R.Tensor(("a // 2", "b * 2"), dtype="float32"):
        a = T.int64()
        b = T.int64()
        gv: R.Tensor((a // 2, b * 2), dtype="float32") = R.reshape(x, R.shape([a // 2, b * 2]))
        return gv