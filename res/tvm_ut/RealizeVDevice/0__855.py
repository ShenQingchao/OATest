# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    I.module_attrs({"attr": 10})
    I.module_global_infos({"vdevice": [I.vdevice({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-unknown-linux-gnu", "tag": ""}, 0, "global"), I.vdevice({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}, 0, "global"), I.vdevice({"keys": ["metal", "gpu"], "kind": "metal", "max_function_args": 31, "max_num_threads": 256, "max_shared_memory_per_block": 32768, "max_threads_per_block": 256, "tag": "", "thread_warp_size": 16}, 0, "global"), I.vdevice({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}, 0, "global")]})
    @R.function
    def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32"), z: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32", vdevice="llvm:0"):
        with R.dataflow():
            x1: R.Tensor((2, 3), dtype="float32") = x
            y1: R.Tensor((2, 3), dtype="float32") = y
            x2: R.Tensor((2, 3), dtype="float32") = x1
            y2: R.Tensor((2, 3), dtype="float32") = y1
            lv0: R.Tensor((2, 3), dtype="float32", vdevice="llvm:0") = R.add(x2, y2)
            gv: R.Tensor((2, 3), dtype="float32", vdevice="llvm:0") = R.multiply(lv0, z)
            R.output(gv)
        return gv