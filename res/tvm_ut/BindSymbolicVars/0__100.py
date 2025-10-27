# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("batch", "m"), dtype="float32"), w0: R.Tensor(("m", "n"), dtype="float32"), w1: R.Tensor(("k", 10), dtype="float32")) -> R.Tensor(("batch", "k"), dtype="float32"):
        batch = T.int64()
        k = T.int64()
        m = T.int64()
        n = T.int64()
        with R.dataflow():
            lv0 = R.call_dps_packed("test0", (x, w0), out_sinfo=R.Tensor((batch, n), dtype="float32"))
            out = R.call_dps_packed("test1", (lv0, w1), out_sinfo=R.Tensor((batch, k), dtype="float32"))
            R.output(out)
        return out