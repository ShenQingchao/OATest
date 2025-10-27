# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def addone(a: T.handle, b: T.handle):
        n, m = T.int32(), T.int32()
        A = T.match_buffer(a, (n, m))
        B = T.match_buffer(b, (n, m))
        # with T.block("root"):
        for i, j in T.grid(n, m):
            with T.block("addone"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] + T.float32(1)

    @T.prim_func
    def sub(A: T.Buffer((16, 16), "float32"), B: T.Buffer((16, 16), "float32"), C: T.Buffer((16, 16), "float32")):
        # with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("sub"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj], B[vi, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = A[vi, vj] - B[vi, vj]

    @R.function
    def main(c0: R.Tensor((16, 16), dtype="float32"), x: R.Tensor(dtype="float32", ndim=2)) -> R.Tuple(R.Tensor(dtype="float32", ndim=2), R.Tensor((16, 16), dtype="float32")):
        n = T.int64()
        m = T.int64()
        cls = Module
        x0: R.Tensor((n, m), dtype="float32") = R.match_cast(x, R.Tensor((n, m), dtype="float32"))
        lv0 = R.call_tir(cls.addone, (c0,), out_sinfo=R.Tensor((n, 16), dtype="float32"))
        lv1 = R.call_tir(cls.addone, (c0,), out_sinfo=R.Tensor((16, 16), dtype="float32"))
        lv2 = R.call_tir(cls.sub, (c0, lv1), out_sinfo=R.Tensor((16, 16), dtype="float32"))
        lv3 = R.call_tir(cls.sub, (lv2, x), out_sinfo=R.Tensor((16, 16), dtype="float32"))
        return (lv0, lv3)