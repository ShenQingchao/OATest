# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(inp: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((2, 2), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((2, 2), dtype="float32") = R.call_pure_packed("my_func1", inp, R.prim_value(0), sinfo_args=(R.Tensor((2, 2), dtype="float32"),))
            lv1: R.Tensor((2, 2), dtype="float32") = R.call_pure_packed("my_func2", lv, R.str("str"), sinfo_args=(R.Tensor((2, 2), dtype="float32"),))
            gv: R.Tensor((2, 2), dtype="float32") = R.call_pure_packed("my_func3", lv1, R.dtype("float32"), sinfo_args=(R.Tensor((2, 2), dtype="float32"),))
            R.output(gv)
        return gv