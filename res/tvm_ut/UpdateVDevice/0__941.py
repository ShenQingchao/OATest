# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    I.module_attrs({"attr": 10})
    I.module_global_infos({"vdevice": [I.vdevice({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-unknown-linux-gnu", "tag": ""}, 0, "global"), I.vdevice({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}, 0, "global"), I.vdevice({"keys": ["metal", "gpu"], "kind": "metal", "max_function_args": 31, "max_num_threads": 256, "max_shared_memory_per_block": 32768, "max_threads_per_block": 256, "tag": "", "thread_warp_size": 16}, 0, "global"), I.vdevice({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}, 0, "global")]})
    @R.function
    def main(a: R.Tensor((128, 128), dtype="float32", vdevice="cuda:1"), c: R.Tensor((128, 128), dtype="float32", vdevice="cuda:1")) -> R.Tensor((128, 128), dtype="float32", vdevice="cuda:1"):
        s: R.Tensor((128, 128), dtype="float32", vdevice="cuda:1") = R.add(a, c)
        return s