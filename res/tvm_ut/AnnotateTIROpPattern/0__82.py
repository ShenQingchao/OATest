# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def broadcast(A: T.Buffer((16, 16), "float32"), B: T.Buffer((16, 16, 16, 16), "float32")):
        T.func_attr({"global_symbol": "elemwise"})
        # with T.block("root"):
        for i0, j0, i1, j1 in T.grid(16, 16, 16, 16):
            with T.block("matmul"):
                vi0, vj0, vi1, vj1 = T.axis.remap("SSSS", [i0, j0, i1, j1])
                T.reads(A[vj0, vj1])
                T.writes(B[vi0, vj0, vi1, vj1])
                B[vi0, vj0, vi1, vj1] = A[vj0, vj1]