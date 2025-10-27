import os
import sys
import time
import random
from datetime import datetime
import shutil
import re
import tvm

import fuzz_utils
import irs_utils
import synthesis_mode
import test_tvm
import get_pass_level
import parse_utils, transform_decompile


def _sample_n_seeds(seed_pool, num, target_cluster='default'):
    cluster_seed_pair_list = []
    all_cluster_name_list = list(seed_pool.keys())
    if target_cluster and target_cluster in all_cluster_name_list:
        num = min(num, len(seed_pool[target_cluster]))
        seed_list = random.sample(seed_pool[target_cluster], num)
        cluster_seed_pair_list = [(target_cluster, seed) for seed in seed_list]
    else:
        for i in range(num):
            cluster_name = random.choice(all_cluster_name_list)
            seed = random.choice(seed_pool[cluster_name])
            cluster_seed_pair_list.append((cluster_name, seed))
    return cluster_seed_pair_list


class Fuzzer:
    def __init__(self,
                 donor_dir,
                 base_irs_dir,
                 execution_time,
                 log_file,
                 failure_dir,
                 execution_dir,
                 fuzz_mod="random"):
        self.DonorPool = irs_utils.IRsPool(donor_dir)
        self.BaseIRsPool = irs_utils.IRsPool(base_irs_dir, max_num=4000)
        self.log_file = log_file
        self.failure_dir = failure_dir
        self.execution_dir = execution_dir
        self.start_time = time.time()
        self.execution_time = fuzz_utils.parse_execution_time(execution_time)
        self.fuzz_mod = fuzz_mod
        os.makedirs(self.failure_dir, exist_ok=True)
        os.makedirs(self.execution_dir, exist_ok=True)

    def log_bug(self, test_case_path, stdout, stderr):
        with open(self.log_file, 'a') as f:
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Test Case Path: {test_case_path}\n")
            # f.write(f"STDOUT: {stdout}\n")
            f.write(f"STDERR: {stderr}\n")
            f.write("=" * 66 + "\n")
        shutil.copy(test_case_path, self.failure_dir)

    def fuzz(self):
        total_mutant_num = 0
        invalid_mutant_num = 0
        all_detected_unique_bugs = []
        performance_bugs_num = 0

        all_base_passes_cluster = self.BaseIRsPool.irs_pool.keys()
        all_donor_passes_cluster = self.DonorPool.irs_pool.keys()
        # print(all_base_passes_cluster)
        # print(all_donor_passes_cluster)
        if len(all_base_passes_cluster) == 0 or len(all_donor_passes_cluster) == 0:
            raise FileNotFoundError("Cannot find irs in the given path!")
        while True:
            for this_donor_pass_cluster in all_donor_passes_cluster:
                donor_ir_list = self.DonorPool.irs_pool[this_donor_pass_cluster]
                for ir_id, donor_ir_clazz in enumerate(donor_ir_list):
                    print(f"[INFO]: Use the {ir_id}th seed IRs:")
                    if time.time() - self.start_time > self.execution_time:
                        print(f"[INFO]: Total generated tests number is: {total_mutant_num};"
                              f"Invalid tests number is:{invalid_mutant_num}; "
                              f"Valid rate is: {1 - invalid_mutant_num / total_mutant_num}")
                        print(f"[INFO]: Total detected unique bugs number is:{len(all_detected_unique_bugs)}")
                        print(f"[INFO]: Total detected performance bugs is:{performance_bugs_num}")
                        print("[INFO]: Finished ALL && Timeout!")
                        return True

                    # --- for each donor ---
                    donor_related_passes = this_donor_pass_cluster.split('__')
                    donor_pass_level = get_pass_level.get_level(donor_related_passes)
                    donor_ir = donor_ir_clazz.ir

                    if donor_pass_level == 'block':
                        try:
                            new_donor_ir = parse_utils.update_irs_inner_fun_name(donor_ir)
                            new_donor_ir = transform_decompile.decompile_ir(new_donor_ir)
                        except Exception as e:
                            print("[ERROR]: Cannot decompile the donor IR from str:\n", e)
                            print(f"[DEBUG]: Seed IR:\n{donor_ir_clazz.content}\n{'*' * 66}")
                            donor_pass_level = 'dataflow'
                        else:
                            donor_ir = new_donor_ir

                    # --- for each seed ---
                    ideal_base_pass_cluster = this_donor_pass_cluster if self.fuzz_mod == "same_pass" else None
                    cluster_seed_pair_list = _sample_n_seeds(self.BaseIRsPool.irs_pool, 10, ideal_base_pass_cluster)
                    # print(cluster_seed_pair_list)
                    for this_base_pass_cluster, base_seed_clazz in cluster_seed_pair_list:
                        all_related_passes = donor_related_passes + this_base_pass_cluster.split('__')
                        all_related_passes = [item for item in all_related_passes if item != 'default']
                        base_seed = base_seed_clazz.ir

                        print(f"[INFO]: {'==' * 66}")
                        print(f"[INFO]: Passes: {all_related_passes}; Pass granularity: {donor_pass_level}")
                        print(f"[INFO]: Donor: {donor_ir_clazz.filename}; Seed:{base_seed_clazz.filename}.")

                        total_mutant_num += 1
                        synthesis_res = synthesis_mode.synthesize(base_seed, donor_ir, donor_pass_level)
                        if not synthesis_res:  # synthesize failed
                            if synthesis_res is False:   # synthesize an invalid test
                                invalid_mutant_num += 1
                            elif synthesis_res is None:  # synthesize nothing
                                total_mutant_num -= 1
                                break
                            continue

                        new_ir_str, model_inputs_list, mod_input_func_name = synthesis_res
                        # print(model_inputs_list)

                        # save the new ir and new test into a file separately.
                        new_ir_str_runnable = re.sub(r'^# from tvm.script', 'from tvm.script', new_ir_str, flags=re.MULTILINE)
                        new_ir_str_runnable = re.sub(r'^metadata = tvm', 'import tvm\nmetadata = tvm', new_ir_str_runnable)

                        new_ir_file_name_short = f"{total_mutant_num}.py"
                        new_ir_file_path = fuzz_utils.save_test_case(self.execution_dir,
                                                                     new_ir_file_name_short,
                                                                     new_ir_str_runnable)
                        print(new_ir_file_path)
                        new_test_path = test_tvm.gen_tvm_test_file(new_ir_str_runnable,
                                                                   all_related_passes,
                                                                   new_ir_file_path,
                                                                   model_inputs_list,
                                                                   mod_input_func_name)

                        # run the test
                        ret_code, stdout, stderr = test_tvm.run_test(new_test_path)
                        base_seed_clazz.usage_count += 1

                        if ret_code != 0:
                            if "not implemented" in stderr or "support" in stderr:  # skip the unsupported
                                continue
                            if 'Cannot find PackedFunc' in stderr:
                                continue
                            if "unsatisfied constraint" in stderr:  # todo: correct the invalid inputs
                                continue
                            if "Memory verification failed" in stderr:
                                continue
                            if "Undefined variable: metadata" in stderr:
                                assert False, str(stderr)
                            if "AssertionError:" in stderr and "alloc_storage" in new_ir_str_runnable:  # skip the UB
                                continue

                            stderr = fuzz_utils.simple_crash_message(stderr)
                            if "[Performance Bug]:" in stderr:
                                performance_bugs_num += 1
                                self.log_bug(new_test_path, stdout, stderr)
                            else:
                                unique_bug_mess = fuzz_utils.extract_unique_crash_message(stderr)
                                if unique_bug_mess not in all_detected_unique_bugs:
                                    all_detected_unique_bugs.append(unique_bug_mess)
                                    self.log_bug(new_test_path, stdout, stderr)
                        else:
                            new_ir = tvm.script.from_source(new_ir_str)
                            self.BaseIRsPool.add_irs(irs_utils.IRs(new_ir_file_path, new_ir_str, new_ir), "__".join(all_related_passes))
                            # continue
            print("[INFO] Finish generating a mutation for each pass group!")


if __name__ == "__main__":
    # if len(sys.argv) < 3 or len(sys.argv) > 6:
    #     print("Usage: python fuzzer.py <donor_dir> <execution_time> [<log_file> [<failure_dir> [<execution_dir>]]]")
    #     sys.exit(1)
    res_dir = '../res'

    donor_dir = sys.argv[1] if len(sys.argv) > 1 else f"{res_dir}/irs_ut_pickle"

    # base_irs_dir = sys.argv[2] if len(sys.argv) > 2 else f"{res_dir}/tvm_ut"
    base_irs_dir = sys.argv[2] if len(sys.argv) > 2 else f"{res_dir}/irs_nnsmith"
    # base_irs_dir = sys.argv[2] if len(sys.argv) > 2 else f"{res_dir}/irs_hirgen"

    execution_time = sys.argv[3] if len(sys.argv) > 3 else "12h"
    log_file = sys.argv[4] if len(sys.argv) > 4 else f"{res_dir}/res_fuzzer_log.txt"
    failure_dir = sys.argv[5] if len(sys.argv) > 5 else f"{res_dir}/res_failures"
    execution_dir = sys.argv[6] if len(sys.argv) > 6 else f"{res_dir}/res_executions"
    fuzz_mod = sys.argv[7] if len(sys.argv) > 7 else "random"  # [random, same_pass]

    fuzzer = Fuzzer(donor_dir=donor_dir,
                    base_irs_dir=base_irs_dir,
                    execution_time=execution_time,
                    log_file=log_file,
                    failure_dir=failure_dir,
                    execution_dir=execution_dir)
    fuzzer.fuzz()
