# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Shape(["batch", "m"]), w0: R.Shape(["m", "n"]), w1: R.Shape(["k", 10])) -> R.Shape(["batch", "k"]):
        batch = T.int64()
        k = T.int64()
        m = T.int64()
        n = T.int64()
        with R.dataflow():
            lv0 = R.call_dps_packed("test0", (x, w0), out_sinfo=R.Tensor((batch, n)))
            out = R.call_dps_packed("test1", (lv0, w1), out_sinfo=R.Tensor((batch, k)))
            R.output(out)
        return out