# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(pure=False)
    def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
        with R.dataflow():
            lv0: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
            lv1: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
            gv1: R.Tensor((2, 3), dtype="float32") = R.multiply(lv0, lv1)
            R.output(gv1)
        R.print(format=R.str("Prevent dataflow block merging"))
        with R.dataflow():
            lv2: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
            lv3: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
            gv2: R.Tensor((2, 3), dtype="float32") = R.multiply(lv2, lv3)
            R.output(gv2)
        gv3: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
        gv4: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
        gv5: R.Tensor((2, 3), dtype="float32") = R.multiply(gv3, gv4)
        gv4_1: R.Tensor((2, 3), dtype="float32") = R.add(gv1, gv2)
        output: R.Tensor((2, 3), dtype="float32") = R.add(gv4_1, gv5)
        return output