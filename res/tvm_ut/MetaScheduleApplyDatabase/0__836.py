# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((2,), "float32"), B: T.Buffer((2,), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i in range(2):
            with T.block("renamed_block"):
                vi = T.axis.spatial(2, i)
                T.reads(A[vi])
                T.writes(B[vi])
                B[vi] = A[vi]