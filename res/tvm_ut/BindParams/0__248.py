# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def func(A: T.Buffer((2, 3), "float32"), B: T.Buffer((3, 2), "float32")):
        # with T.block("root"):
        for i, j in T.grid(3, 2):
            with T.block("transpose"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vj, vi])
                T.writes(B[vi, vj])
                B[vi, vj] = A[vj, vi]

    @R.function
    def main(c0: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((3, 2), dtype="float32"):
        cls = Module
        lv0 = R.call_tir(cls.func, (c0,), out_sinfo=R.Tensor((3, 2), dtype="float32"))
        return lv0