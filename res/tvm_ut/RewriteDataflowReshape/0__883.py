# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((8, 8), dtype="float16")) -> R.Tensor((8, 8), dtype="float16"):
        with R.dataflow():
            gv: R.Tensor((8, 8), dtype="float16") = R.call_pure_packed("foo", x, x, sinfo_args=(R.Tensor((8, 8), dtype="float16"),))
            out: R.Tensor((8, 8), dtype="float16") = R.call_pure_packed("foo", gv, gv, sinfo_args=(R.Tensor((8, 8), dtype="float16"),))
            R.output(out)
        return out