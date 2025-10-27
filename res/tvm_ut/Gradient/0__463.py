# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            lv1: R.Tensor((3, 3), dtype="float32") = R.matmul(x, y, out_dtype="void")
            lv2: R.Tensor((3, 3), dtype="float32") = R.permute_dims(x, axes=None)
            lv3: R.Tensor((3, 3), dtype="float32") = R.matmul(lv2, y, out_dtype="void")
            lv4: R.Tensor((3, 3), dtype="float32") = R.permute_dims(y, axes=None)
            lv5: R.Tensor((3, 3), dtype="float32") = R.matmul(x, lv4, out_dtype="void")
            lv6: R.Tensor((3, 3), dtype="float32") = R.permute_dims(x, axes=None)
            lv7: R.Tensor((3, 3), dtype="float32") = R.permute_dims(y, axes=None)
            lv8: R.Tensor((3, 3), dtype="float32") = R.matmul(lv6, lv7, out_dtype="void")
            lv8_1: R.Tensor((3, 3), dtype="float32") = R.add(lv1, lv3)
            lv9: R.Tensor((3, 3), dtype="float32") = R.add(lv8_1, lv5)
            lv9_1: R.Tensor((3, 3), dtype="float32") = R.add(lv9, lv8)
            gv: R.Tensor((), dtype="float32") = R.sum(lv9_1, axis=None, keepdims=False)
            R.output(gv)
        return gv