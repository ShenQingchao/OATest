# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((32, 8), dtype="int32"), y: R.Tensor((8, 8), dtype="int32"), bias: R.Tensor((8,), dtype="int32")) -> R.Tensor((32, 8), dtype="int32"):
        with R.dataflow():
            lv0: R.Tensor((32, 8), dtype="int32") = R.matmul(x, y, out_dtype="int32")
            lv1: R.Tensor((32, 8), dtype="int32") = R.add(lv0, bias)
            lv2: R.Tensor((32, 8), dtype="int32") = R.clip(lv1, R.prim_value(-128), R.prim_value(127))
            R.output(lv2)
        return lv2