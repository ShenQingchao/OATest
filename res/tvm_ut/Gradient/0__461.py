# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def MLP(x: R.Tensor((4, 5), dtype="float32"), w_0: R.Tensor((5, 5), dtype="float32"), w_1: R.Tensor((5, 5), dtype="float32"), w_2: R.Tensor((5, 5), dtype="float32"), b_0: R.Tensor((5,), dtype="float32"), b_1: R.Tensor((5,), dtype="float32"), b_2: R.Tensor((5,), dtype="float32"), y: R.Tensor((4,), dtype="int64")) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((4, 5), dtype="float32") = R.matmul(x, w_0, out_dtype="void")
            lv1: R.Tensor((4, 5), dtype="float32") = R.add(lv, b_0)
            lv2: R.Tensor((4, 5), dtype="float32") = R.nn.relu(lv1)
            lv3: R.Tensor((4, 5), dtype="float32") = R.matmul(lv2, w_1, out_dtype="void")
            lv4: R.Tensor((4, 5), dtype="float32") = R.add(lv3, b_1)
            lv5: R.Tensor((4, 5), dtype="float32") = R.nn.relu(lv4)
            lv6: R.Tensor((4, 5), dtype="float32") = R.matmul(lv5, w_2, out_dtype="void")
            lv7: R.Tensor((4, 5), dtype="float32") = R.add(lv6, b_2)
            lv8: R.Tensor((4, 5), dtype="float32") = lv7
            lv9: R.Tensor((4, 5), dtype="float32") = R.nn.log_softmax(lv8, axis=-1)
            lv10: R.Tensor((), dtype="float32") = R.nn.nll_loss(lv9, y, reduction="mean", ignore_index=-100)
            gv: R.Tensor((), dtype="float32") = lv10
            R.output(gv)
        return gv