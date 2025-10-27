# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    I.module_attrs({"attr": 10})
    I.module_global_infos({"vdevice": [I.vdevice({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-unknown-linux-gnu", "tag": ""}, 0, "global"), I.vdevice({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}, 0, "global"), I.vdevice({"keys": ["metal", "gpu"], "kind": "metal", "max_function_args": 31, "max_num_threads": 256, "max_shared_memory_per_block": 32768, "max_threads_per_block": 256, "tag": "", "thread_warp_size": 16}, 0, "global"), I.vdevice({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}, 0, "global")]})
    @R.function
    def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32"), z: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32", vdevice="cuda:0"):
        with R.dataflow():
            lv0: R.Tensor((2, 3), dtype="float32", vdevice="llvm:0") = R.add(x, y)
            lv1: R.Tensor((2, 3), dtype="float32", vdevice="cuda:0") = R.to_vdevice(lv0, dst_vdevice="cuda:0")
            lv2: R.Tensor((2, 3), dtype="float32") = R.add(z, z)
            gv: R.Tensor((2, 3), dtype="float32", vdevice="cuda:0") = R.multiply(lv1, lv2)
            R.output(gv)
        return gv