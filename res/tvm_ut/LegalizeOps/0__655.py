# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((4, 4), dtype="float32"), indices: R.Tensor((2, 2), dtype="int64"), updates: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((4, 4), dtype="float32"):
        gv: R.Tensor((4, 4), dtype="float32") = R.scatter_elements(x, indices, updates, axis=1, reduction="update")
        return gv