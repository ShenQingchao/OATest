# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def unused_func1(x: T.Buffer((16, 16), "float32"), y: T.Buffer((16, 16), "float32"), z: T.Buffer((16, 16), "float32")):
        T.func_attr({"global_symbol": "tir_unused"})
        # with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("add"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(x[vi, vj], y[vi, vj])
                T.writes(z[vi, vj])
                z[vi, vj] = x[vi, vj] + y[vi, vj]

    @R.function(private=True)
    def unused_func2(x: R.Tensor((16, 16), dtype="float32"), w: R.Tensor((16, 16), dtype="float32")) -> R.Tensor((16, 16), dtype="float32"):
        gv0: R.Tensor((16, 16), dtype="float32") = R.add(x, w)
        return gv0

    @R.function
    def main(x: R.Tensor((16, 16), dtype="float32"), w: R.Tensor((16, 16), dtype="float32")) -> R.Tensor((16, 16), dtype="float32"):
        gv0: R.Tensor((16, 16), dtype="float32") = R.add(x, w)
        return gv0