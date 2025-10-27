# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def tir_matmul(x: T.handle, y: T.handle, z: T.handle):
        m, n = T.int32(), T.int32()
        A = T.match_buffer(x, (m, n), "float16")
        k = T.int32()
        B = T.match_buffer(y, (n, k), "float16")
        C = T.match_buffer(z, (m, k))
        # with T.block("root"):
        for i, j, k_1 in T.grid(m, k, n):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k_1])
                T.reads(A[vi, vk], B[vk, vj])
                T.writes(C[vi, vj])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + T.Cast("float32", A[vi, vk] * B[vk, vj])