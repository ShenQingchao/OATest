# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def exp(A: T.Buffer((2, 3), "float32"), B: T.Buffer((2, 3), "float32")):
        T.func_attr({"op_pattern": 0})
        T.evaluate(0)

    @R.function
    def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")):
        cls = Module
        with R.dataflow():
            a = R.call_tir(cls.exp, (x,), out_sinfo=R.Tensor((2, 3), dtype="float32"))
            b = R.call_tir(cls.exp, (a,), out_sinfo=R.Tensor((2, 3), dtype="float32"))
            c = R.call_dps_packed("packed_dps", (a,), out_sinfo=R.Tensor((2, 3), dtype="float32"))
            R.output(b, c)
        return (b, c)