# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def injective(A: T.Buffer((4, 4, 4, 4), "float32"), B: T.Buffer((16, 16), "float32")):
        T.func_attr({"global_symbol": "elemwise"})
        # with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("matmul"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi // 4, vj // 4, vi % 4, vj % 4])
                T.writes(B[vi, vj])
                B[vi, vj] = A[vi // 4, vj // 4, vi % 4, vj % 4]