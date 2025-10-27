# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    I.module_global_infos({"vdevice": [I.vdevice({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-unknown-linux-gnu", "tag": ""}, 0, "global")]})
    @R.function
    def func_cuda(A: R.Tensor((32, 32), dtype="float32"), B: R.Tensor((32, 32), dtype="float32")) -> R.Tensor((32, 32), dtype="float32"):
        C: R.Tensor((32, 32), dtype="float32") = R.add(A, B)
        return C

    @R.function
    def func_llvm(A: R.Tensor((32, 32), dtype="float32", vdevice="llvm:0"), B: R.Tensor((32, 32), dtype="float32", vdevice="llvm:0")) -> R.Tensor((32, 32), dtype="float32", vdevice="llvm:0"):
        C: R.Tensor((32, 32), dtype="float32", vdevice="llvm:0") = R.add(A, B)
        return C