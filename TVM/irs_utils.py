import os
import tvm
import pickle


def deserialize_irs(filepath="ut_irs.pkl"):
    with open(filepath, 'rb') as file:
        res = pickle.load(file)
    return res


def load_pickle(filepath):
    try:
        pass_name, pass_call, irs = None, None, None
        with open(filepath, 'rb') as file:
            res_pk = pickle.load(file)
        pass_name = res_pk['pass_name']
        pass_args = res_pk['pass_args']
        pass_kwargs = res_pk['pass_kwargs']
        # print(pass_name, pass_args)
        pass_api = eval(f"tvm.relax.transform.{pass_name}")
        pass_call = pass_api(*pass_args, **pass_kwargs)
        irs = res_pk['mod']
    except Exception as e:
        pass_name = pass_name if pass_name else None
        irs = irs if irs else None
        pass_call = pass_call if pass_call else None
        if not pass_name or not irs:
            os.remove(filepath)
            return None
    return pass_name, pass_call, irs


class IRs:
    def __init__(self, filename, content, ir, pass_name, pass_call=None):
        self.filename = filename
        self.content = content
        self.ir = ir
        self.pass_call = pass_call
        self.pass_name = pass_name
        self.usage_count = 0
        self.score = 0


class IRsPool:
    def __init__(self, seed_folder, max_num=2000):
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
                if file_path.endswith(".pkl"):
                    load_res = load_pickle(file_path)
                    if not load_res:
                        continue
                    _pass, pass_call, content = load_res

                elif file_path.endswith('.py'):
                    _pass = os.path.basename(root)
                    pass_call = None
                    with open(file_path, 'r') as f:
                        content = f.read()
                try:
                    ir = tvm.script.from_source(content)
                except Exception as e:
                    print(e)
                    # os.remove(file_path)
                    continue
                if _pass not in self.irs_pool.keys():
                    self.irs_pool[_pass] = []
                self.irs_pool[_pass].append(IRs(file_path, content, ir, _pass, pass_call))
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
    import tvm
    pool = IRsPool("../res/irs_ut_pickle_2k")
    for k, v in pool.irs_pool.items():
        for base_seed_clazz in v:
            # base_seed = tvm.script.from_source(base_seed_clazz.content)
            # print(base_seed_clazz.pass_name)
            continue
