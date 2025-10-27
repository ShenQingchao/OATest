# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("batch_size", 1, 16), dtype="float32"), weight_table: R.Tensor((16, "weight_table_size"), dtype="float32"), routing_table: R.Tensor((32,), dtype="int64")) -> R.Tensor(("batch_size", 1, 32), dtype="float32"):
        batch_size = T.int64()
        weight_table_size = T.int64()
        with R.dataflow():
            weight: R.Tensor((16, 32), dtype="float32") = R.take(weight_table, routing_table, axis=1)
            out: R.Tensor((batch_size, 1, 32), dtype="float32") = R.matmul(x, weight, out_dtype="void")
            R.output(out)
        return out