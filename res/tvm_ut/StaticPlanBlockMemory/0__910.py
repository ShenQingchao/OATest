# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def cumsum(var_A: T.handle, var_A_1: T.handle, var_exclusive_scan_thrust: T.handle):
        T.evaluate(0)

    @R.function
    def main(probs: R.Tensor(("batch_size", "vocab_size"), dtype="float32")) -> R.Tensor(("batch_size", "vocab_size"), dtype="float32"):
        batch_size = T.int64()
        vocab_size = T.int64()
        R.func_attr({"relax.force_pure": 1, "relax.memory_plan_dynamic_func_output": 1, "tir_non_negative_var": ["vocab_size"], "tir_var_upper_bound": {"batch_size": 32}})
        cls = Module
        lv1: R.Tensor((2 * (batch_size * vocab_size * 4) + 4194304,), dtype="uint8") = R.builtin.alloc_tensor(R.shape([2 * (batch_size * vocab_size * 4) + 4194304]), R.dtype("uint8"), R.prim_value(0), R.str("global"))
        alloc1: R.Tensor((batch_size, vocab_size), dtype="float32") = R.builtin.alloc_tensor(R.shape([batch_size, vocab_size]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        cls.cumsum(probs, lv1, alloc1)
        cumsum: R.Tensor((batch_size, vocab_size), dtype="float32") = alloc1
        lv1_1: R.Tensor((batch_size, vocab_size), dtype="int32") = R.call_packed("vm.builtin.reshape", cumsum, R.shape([batch_size, vocab_size]), sinfo_args=(R.Tensor((batch_size, vocab_size), dtype="float32"),))
        return lv1_1