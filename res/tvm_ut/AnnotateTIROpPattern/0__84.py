# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def tir_bias_add(A: T.Buffer((1, 1000), "float32"), B: T.Buffer((1000,), "float32"), C: T.Buffer((1, 1000), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(1, 1000):
            with T.block("T_add"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(A[ax0, ax1], B[ax1])
                T.writes(C[ax0, ax1])
                C[ax0, ax1] = A[ax0, ax1] + B[ax1]