# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def addone(A: T.Buffer((16, 16), "int32"), B: T.Buffer((16, 16), "int32")):
        # with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("addone"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] + 1

    @R.function
    def main(c0: R.Tensor((16, 16), dtype="int32")) -> R.Tensor((16, 16), dtype="int32"):
        cls = Module
        lv0 = R.call_tir(cls.addone, (c0,), out_sinfo=R.Tensor((16, 16), dtype="int32"))
        return lv0