# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def sum_1d(X_handle: T.handle, Sum: T.Buffer((T.int64(1),), "float32")):
        num_elements = T.int64()
        X = T.match_buffer(X_handle, (num_elements,))
        # with T.block("root"):
        for i in range(num_elements):
            with T.block("sum"):
                vi = T.axis.reduce(num_elements, i)
                T.reads(X[vi])
                T.writes(Sum[0])
                with T.init():
                    Sum[0] = T.float32(0)
                Sum[0] = Sum[0] + X[vi]

    @T.prim_func(private=True)
    def sum_scalar(X: T.Buffer((T.int64(1),), "float32"), Y: T.Buffer((T.int64(1),), "float32"), Sum: T.Buffer((T.int64(1),), "float32")):
        # with T.block("root"):
        for i in range(T.int64(1)):
            with T.block("Out"):
                vi = T.axis.spatial(T.int64(1), i)
                T.reads(X[vi], Y[vi])
                T.writes(Sum[vi])
                Sum[vi] = X[vi] + Y[vi]

    @R.function(private=True)
    def fused(x: R.Tensor((64,), dtype="float32"), y: R.Tensor((16,), dtype="float32")) -> R.Tensor((1,), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            x_sum = R.call_tir(cls.sum_1d, (x,), out_sinfo=R.Tensor((1,), dtype="float32"))
            y_sum = R.call_tir(cls.sum_1d, (y,), out_sinfo=R.Tensor((1,), dtype="float32"))
            gv = R.call_tir(cls.sum_scalar, (x_sum, y_sum), out_sinfo=R.Tensor((1,), dtype="float32"))
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((64,), dtype="float32"), y: R.Tensor((16,), dtype="float32")) -> R.Tensor((1,), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv: R.Tensor((1,), dtype="float32") = cls.fused(x, y)
            R.output(gv)
        return gv