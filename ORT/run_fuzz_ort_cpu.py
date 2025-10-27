import os
import sys
import time
import random
from datetime import datetime
import shutil
import multiprocessing
import subprocess
import onnx

import irs_utils
import fuzz_utils
import synthesis_mode
import get_pass_level

env = os.environ.copy()
env["GCOV_FLUSH_INTERVAL"] = "1"


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


def run_test(model_path):
    process = subprocess.Popen(
        ["python", "test_ort.py", model_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode(), stderr.decode()


def collect_cov(cov_cnt):
    ort_cov_dir = "/software/onnxruntime/build/Linux/RelWithDebInfo/CMakeFiles/onnxruntime_optimizer.dir/software/onnxruntime/onnxruntime/"
    cov_collect = f"lcov --capture --directory {ort_cov_dir} " \
                  f"--output-file ./sp_ORT_ModelTailor_{cov_cnt}.info " \
                  f"--rc lcov_branch_coverage=1"
    subprocess.run(cov_collect, shell=True)


class Fuzzer:
    def __init__(self,
                 donor_dir,
                 base_irs_dir,
                 execution_time,
                 log_file,
                 failure_dir,
                 execution_dir,
                 fuzz_mod="random"):
        self.DonorPool = irs_utils.IRsPool(donor_dir, max_num=5000)
        self.BaseIRsPool = irs_utils.IRsPool(base_irs_dir, max_num=4000)  # , max_num=1000
        self.log_file = log_file
        self.failure_dir = failure_dir
        self.execution_dir = execution_dir
        self.start_time = time.time()
        self.execution_time = fuzz_utils.parse_execution_time(execution_time)
        self.fuzz_mod = fuzz_mod
        os.makedirs(self.failure_dir, exist_ok=True)
        os.makedirs(self.execution_dir, exist_ok=True)

        self.manager = multiprocessing.Manager()
        self.invalid_mutant_num = multiprocessing.Value('i', 0)
        self.all_detected_unique_bugs = self.manager.list()
        self.performance_bugs_num = multiprocessing.Value('i', 0)

    def log_bug(self, seed_path, donor_path, test_case_path, stderr):
        with open(self.log_file, 'a') as f:
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Test Case Path: {test_case_path}\n")
            f.write(f"Seed Path: {seed_path}\n")
            f.write(f"Donor Path: {donor_path}\n")
            f.write(f"ERROR: {stderr}\n")
            f.write("=" * 66 + "\n")
        shutil.copy(test_case_path, self.failure_dir)

    def single_task(self, total_mutant_num, base_seed_clazz, donor_ir_clazz, donor_pass_level, all_related_passes):
        print(f"[INFO]: {'==' * 66}")
        print(f"[INFO]: Passes: {all_related_passes}; Pass granularity: {donor_pass_level}")
        print(f"[INFO]: Donor: {donor_ir_clazz.filename} ; Seed: {base_seed_clazz.filename}")

        synthesis_res = synthesis_mode.synthesize(base_seed_clazz.ir, donor_ir_clazz.ir, donor_pass_level)

        if not synthesis_res:  # synthesize failed
            if synthesis_res is False:   # synthesize an invalid test
                # with open("./invalid_case_cpu.log", "a") as f:
                #     f.write(f"Test Case Path: {total_mutant_num}.onnx\n")
                #     f.write(f"Seed Path: {base_seed_clazz.filename}\n")
                #     f.write(f"Donor Path: {donor_ir_clazz.filename}\n")
                #     f.write("=" * 66 + "\n")
                self.invalid_mutant_num.value += 1
                print(f"[WARNING][INVALID]: Synthesizing failed, Plz check it!")
            return

        new_ir_file_name_short = f"{total_mutant_num}.onnx"
        new_ir_file_path = os.path.join(self.execution_dir, new_ir_file_name_short)
        onnx.save(synthesis_res, new_ir_file_path)
        # print(f"[INFO] Synthesized model are save in {new_ir_file_path}")

        ret_code, _, stderr = run_test(new_ir_file_path)
        if ret_code:  # detected a bug
            if "NOT_IMPLEMENTED" in stderr:
                print(f"[WARNING] SKip the unsupported issue...")
                return
            elif "Unable to handle object of type" in stderr:
                print(f"[WARNING] SKip the FP arising from invalid seed graph...")
                return
            stderr = fuzz_utils.simple_crash_message(stderr)
            unique_bug_mess = fuzz_utils.extract_unique_crash_message(stderr)
            if unique_bug_mess not in self.all_detected_unique_bugs or "Segmentation fault" in unique_bug_mess:
                print(unique_bug_mess)
                self.all_detected_unique_bugs.append(unique_bug_mess)
                self.log_bug(base_seed_clazz.filename, donor_ir_clazz.filename, new_ir_file_path, stderr)
                print(f"[INFO] Detected bug number: {len(self.all_detected_unique_bugs)}")
        else:
            all_pass_cluster_str = "_".join(all_related_passes)
            # new_ir_clazz = irs_utils.IRs(new_ir_file_path, synthesis_res, all_pass_cluster_str, pass_call=None)
            # self.BaseIRsPool.add_irs(new_ir_clazz, all_pass_cluster_str)  # fixme: ...
            print(f"[SUCCESS]: Add the synthesized model {new_ir_file_path} into Seed Pool!")

    def fuzz(self):
        total_mutant_num = 0
        all_base_passes_cluster = self.BaseIRsPool.irs_pool.keys()
        all_donor_passes_cluster = self.DonorPool.irs_pool.keys()
        # print(all_base_passes_cluster)
        # print(all_donor_passes_cluster)
        if len(all_base_passes_cluster) == 0 or len(all_donor_passes_cluster) == 0:
            raise FileNotFoundError("Cannot find IRs in the given path!")

        last_save_cov_time = time.time()
        cov_during_time = 1800
        cov_cnt = 0
        while True:
            for this_donor_pass_cluster in all_donor_passes_cluster:
                donor_ir_list = self.DonorPool.irs_pool[this_donor_pass_cluster]
                for ir_id, donor_ir_clazz in enumerate(donor_ir_list):
                    current_time = time.time()
                    during_time = current_time - self.start_time

                    if during_time > self.execution_time:
                        collect_cov(cov_cnt)
                        print(f"[INFO]: Total generated tests number is: {total_mutant_num};"
                              f"Invalid tests number is:{self.invalid_mutant_num.value}; "
                              f"Valid rate is: {1 - self.invalid_mutant_num.value / total_mutant_num}")
                        print(f"[INFO]: Total detected unique bugs number is:{len(self.all_detected_unique_bugs)}")
                        print(f"[INFO]: Total detected performance bugs is:{self.performance_bugs_num.value}")
                        print("[INFO]: Finished ALL && Timeout!")
                        return True

                    # --- for each donor ---
                    donor_related_passes = this_donor_pass_cluster.split('__')
                    donor_pass_level = get_pass_level.get_level(donor_related_passes)

                    # --- for each seed ---
                    ideal_base_pass_cluster = this_donor_pass_cluster if self.fuzz_mod == "same_pass" else None
                    cluster_seed_pair_list = _sample_n_seeds(self.BaseIRsPool.irs_pool, 5, ideal_base_pass_cluster)
                    # print(cluster_seed_pair_list)
                    all_tasks_list = []
                    processes = []
                    for this_base_pass_cluster, base_seed_clazz in cluster_seed_pair_list:
                        all_related_passes = donor_related_passes + this_base_pass_cluster.split('__')
                        all_related_passes = [item for item in all_related_passes if item != 'default']
                        total_mutant_num += 1
                        all_tasks_list.append([total_mutant_num, base_seed_clazz, donor_ir_clazz, donor_pass_level, all_related_passes])
                        p = multiprocessing.Process(target=self.single_task, args=(total_mutant_num, base_seed_clazz, donor_ir_clazz, donor_pass_level, all_related_passes,))
                        processes.append(p)
                        p.start()
                    for p in processes:
                        p.join()

                    print("Finish All")
                    current_time = time.time()
                    during_time = current_time - last_save_cov_time
                    if during_time > cov_during_time:
                        collect_cov(cov_cnt)
                        cov_cnt += 1
                        last_save_cov_time = current_time
            print("[INFO] Finish generating a mutation for each pass group!")


if __name__ == "__main__":
    res_dir = '../res/CCS25'
    donor_dir = sys.argv[1] if len(sys.argv) > 1 else f"{res_dir}/onnx_ut/onnx_models_gf"

    base_irs_dir_ut = sys.argv[2] if len(sys.argv) > 2 else f"{res_dir}/onnx_ut/onnx_models_gf"
    base_irs_dir_nnsmith = sys.argv[2] if len(sys.argv) > 2 else f"{res_dir}/onnx_nnsmith"
    # base_irs_dir = [base_irs_dir_ut, base_irs_dir_nnsmith]
    base_irs_dir = [base_irs_dir_nnsmith]

    execution_time = sys.argv[3] if len(sys.argv) > 3 else "12h"
    res_dir = os.path.join(res_dir, "onnx_res_1229")
    if not res_dir:
        os.mkdir(res_dir)
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
