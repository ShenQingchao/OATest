# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(inp: R.Tensor((16, 32), dtype="float32")) -> R.Tensor((16, 16), dtype="float32"):
        with R.dataflow():
            tup: R.Tuple(R.Tensor((16, 16), dtype="float32"), R.Tensor((16, 16), dtype="float32")) = R.split(inp, indices_or_sections=[16], axis=1)
            lv1: R.Tensor((16, 16), dtype="float32") = tup[0]
            lv2: R.Tensor((16, 16), dtype="float32") = tup[1]
            out: R.Tensor((16, 16), dtype="float32") = R.add(lv1, lv2)
            R.output(out)
        return out