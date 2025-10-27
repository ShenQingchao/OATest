# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 10, 4), dtype="float32")) -> R.Tuple(R.Tensor((2, 3, 4), dtype="float32"), R.Tensor((2, 4, 4), dtype="float32"), R.Tensor((2, 3, 4), dtype="float32")):
        gv: R.Tuple(R.Tensor((2, 3, 4), dtype="float32"), R.Tensor((2, 4, 4), dtype="float32"), R.Tensor((2, 3, 4), dtype="float32")) = R.split(x, indices_or_sections=[3, 7], axis=1)
        return gv