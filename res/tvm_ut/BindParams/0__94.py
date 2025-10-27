# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def tir_matmul(A: T.Buffer((16, 16), "float32"), B: T.Buffer((16, 16), "float32"), C: T.Buffer((16, 16), "float32")):
        # with T.block("root"):
        for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
            with T.block("matmul"):
                vi = T.axis.spatial(16, i0 * 4 + i1)
                vj = T.axis.spatial(16, j)
                vk = T.axis.reduce(16, k0 * 4 + k1)
                T.reads(A[vi, vk], B[vk, vj])
                T.writes(C[vi, vj])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    @R.function
    def main(x: R.Tensor((16, 16), dtype="float32"), w: R.Tensor((16, 16), dtype="float32")) -> R.Tensor((16, 16), dtype="float32"):
        cls = Module
        gv0 = R.call_tir(cls.tir_matmul, (x, w), out_sinfo=R.Tensor((16, 16), dtype="float32"))
        return gv0