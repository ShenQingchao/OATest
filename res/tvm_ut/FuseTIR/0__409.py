# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def sum_1d(X_handle: T.handle, Y: T.Buffer((T.int64(1),), "float32")):
        num_elements = T.int64()
        X = T.match_buffer(X_handle, (num_elements,))
        # with T.block("root"):
        for i in range(num_elements):
            with T.block("sum"):
                vi = T.axis.reduce(num_elements, i)
                T.reads(X[vi])
                T.writes(Y[0])
                with T.init():
                    Y[0] = T.float32(0)
                Y[0] = Y[0] + X[vi]

    @R.function(private=True)
    def fused(x: R.Tensor((64,), dtype="float32")) -> R.Tensor((1,), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.sum_1d, (x,), out_sinfo=R.Tensor((1,), dtype="float32"))
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((64,), dtype="float32")) -> R.Tensor((1,), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv: R.Tensor((1,), dtype="float32") = cls.fused(x)
            R.output(gv)
        return gv