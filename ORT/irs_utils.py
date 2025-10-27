import os
import onnx
import re


class IRs:
    def __init__(self, filename, ir, pass_name, pass_call=None):
        self.filename = filename
        self.ir = ir
        self.pass_name = pass_name
        self.pass_call = pass_call
        self.usage_count = 0
        self.score = 0


class IRsPool:
    def __init__(self, seed_folder, max_num=10000):
        self.irs_pool = {}
        self.seed_folder = seed_folder
        self.max_pool_num = max_num
        if isinstance(seed_folder, list):  # combined seed pool with multiple sources
            for sub_seed_folder in seed_folder:
                self.load_seed_pool(sub_seed_folder)
        else:
            self.load_seed_pool(seed_folder)

    def load_seed_pool(self, seed_folder):
        seed_cnt = 0
        for root, _, files in os.walk(seed_folder):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.endswith('.onnx'):
                    # print(file_path)
                    # _pass = re.sub(r'_\d+', '', file[:-5])
                    _pass = root.split("/")[-1]
                    pass_call = None
                    try:
                        ir = onnx.load(file_path)
                        onnx.checker.check_model(ir, full_check=True)
                    except Exception as e:
                        print(f"[ERROR]: {e}")
                        continue
                else:
                    print(file_path)
                    continue
                if _pass not in self.irs_pool.keys():
                    self.irs_pool[_pass] = []
                self.irs_pool[_pass].append(IRs(file_path, ir, _pass, pass_call))
                seed_cnt += 1
                if seed_cnt >= self.max_pool_num:
                    print(f"[INFO]: Finish load {seed_cnt} irs from {seed_folder}!")
                    return
        print(f"[INFO]: Finish load {seed_cnt} irs from {seed_folder}!")

    def add_irs(self, new_seed_clazz, this_pass_list_str):
        if this_pass_list_str not in self.irs_pool.keys():
            self.irs_pool[this_pass_list_str] = []
        self.irs_pool[this_pass_list_str].append(new_seed_clazz)
        print(f"[INFO]: Add the synthesis irs to seed pool about:{this_pass_list_str}")


if __name__ == '__main__':
    pool = IRsPool("../res/onnx_ut")
    for k, v in pool.irs_pool.items():
        # print(k, len(v))
        print(f'"{k}": "dataflow",')

