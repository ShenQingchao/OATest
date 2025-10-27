# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def tir_matmul(A: T.Buffer((32, 32), "float32"), B: T.Buffer((32, 32), "float32"), C: T.Buffer((32, 32), "float32")):
        # with T.block("root"):
        for i0, j0, k0 in T.grid(32, 32, 32):
            with T.block(""):
                i, j, k = T.axis.remap("SSR", [i0, j0, k0])
                T.reads(A[i, k], B[j, k])
                T.writes(C[i, j])
                with T.init():
                    C[i, j] = T.float32(0)
                C[i, j] = C[i, j] + A[i, k] * B[j, k]

    @T.prim_func
    def tir_relu(A: T.Buffer((32, 32), "float32"), B: T.Buffer((32, 32), "float32")):
        # with T.block("root"):
        for i, j in T.grid(32, 32):
            with T.block(""):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(B[vi, vj])
                B[vi, vj] = T.max(A[vi, vj], T.float32(0))

    @R.function
    def main(x: R.Tensor((32, 32), dtype="float32"), w: R.Tensor((32, 32), dtype="float32")) -> R.Tensor((32, 32), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv0 = R.call_tir(cls.tir_matmul, (x, w), out_sinfo=R.Tensor((32, 32), dtype="float32"))
            lv1 = R.call_tir(cls.tir_relu, (lv0,), out_sinfo=R.Tensor((32, 32), dtype="float32"))
            R.output(lv1)
        return lv1