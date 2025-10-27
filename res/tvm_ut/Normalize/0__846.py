# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def foo(x: R.Tensor(("m", "n"), dtype="float32")) -> R.Tensor(("m", "n"), dtype="float32"):
        m = T.int64()
        n = T.int64()
        with R.dataflow():
            lv0 = R.call_dps_packed("test.op.identity", (x,), out_sinfo=R.Tensor((m, n), dtype="float32"))
            gv0 = R.call_dps_packed("test.op.identity", (lv0,), out_sinfo=R.Tensor((m, n), dtype="float32"))
            R.output(gv0)
        return gv0