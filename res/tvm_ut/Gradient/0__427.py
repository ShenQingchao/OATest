# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            lv1: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = x, y
            lv2: R.Tensor((3, 3), dtype="float32") = lv1[0]
            lv3: R.Tensor((3, 3), dtype="float32") = R.add(lv2, x)
            lv4: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = lv1
            lv5: R.Tensor((3, 3), dtype="float32") = lv4[0]
            lv6: R.Tensor((3, 3), dtype="float32") = R.add(lv5, lv3)
            gv: R.Tensor((), dtype="float32") = R.sum(lv6, axis=None, keepdims=False)
            R.output(gv)
        return gv