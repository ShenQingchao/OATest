# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def tir_add(x: T.Buffer((16, 16), "float32"), y: T.Buffer((16, 16), "float32"), z: T.Buffer((16, 16), "float32")):
        # with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("add"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(x[vi, vj], y[vi, vj])
                T.writes(z[vi, vj])
                z[vi, vj] = x[vi, vj] + y[vi, vj]

    @R.function(private=True)
    def unused_func(x: R.Tensor(("m", "n"), dtype="float32"), w: R.Tensor(("n", "k"), dtype="float32")) -> R.Tensor(dtype="float32", ndim=2):
        m = T.int64()
        n = T.int64()
        k = T.int64()
        gv0: R.Tensor(dtype="float32", ndim=2) = R.add(x, w)
        return gv0

    @R.function
    def main(x: R.Tensor(("m", "n"), dtype="float32"), w: R.Tensor(("n", "k"), dtype="float32")) -> R.Tensor(("m + 1", "k"), dtype="float32"):
        m = T.int64()
        k = T.int64()
        n = T.int64()
        cls = Module
        gv0 = R.call_tir(cls.tir_add, (x, w), out_sinfo=R.Tensor((m + 1, k), dtype="float32"))
        return gv0