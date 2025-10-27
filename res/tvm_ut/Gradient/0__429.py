# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            lv0: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = x, y
            lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
            lv2: R.Tensor((3, 3), dtype="float32") = lv0[0]
            lv3: R.Tensor((3, 3), dtype="float32") = R.add(lv2, y)
            lv4: R.Tensor((3, 3), dtype="float32") = R.add(lv1, lv3)
            lv5: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = x, y
            lv6: R.Tensor((3, 3), dtype="float32") = lv5[0]
            lv7: R.Tensor((3, 3), dtype="float32") = lv0[0]
            lv8: R.Tensor((3, 3), dtype="float32") = R.add(lv4, lv6)
            lv9: R.Tensor((3, 3), dtype="float32") = R.add(lv8, lv7)
            gv: R.Tensor((), dtype="float32") = R.sum(lv9, axis=None, keepdims=False)
            R.output(gv)
        return gv