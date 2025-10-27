# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
        with R.dataflow():
            y = R.call_dps_packed("func_packed_dps", (x,), out_sinfo=R.Tensor((2, 3), dtype="float32"))
            R.output(y)
        return y