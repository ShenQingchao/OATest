# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), y: R.Tensor((3, 3), dtype="float32"), z: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            lv1: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = y, z
            lv2: R.Tensor((3, 3), dtype="float32") = x[0]
            lv3: R.Tensor((3, 3), dtype="float32") = lv1[0]
            lv4: R.Tensor((3, 3), dtype="float32") = R.add(lv2, lv3)
            gv: R.Tensor((), dtype="float32") = R.sum(lv4, axis=None, keepdims=False)
            R.output(gv)
        return gv