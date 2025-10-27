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



def run_test(model_path):
    process = subprocess.Popen(
        ["python", "test_ort.py", model_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode(), stderr.decode()


def collect_cov(cov_dir="."):
    ort_cov_dir = "/software/onnxruntime/build/Linux/RelWithDebInfo/CMakeFiles/onnxruntime_optimizer.dir/software/onnxruntime/onnxruntime/"
    cov_collect = f"lcov --capture --directory {ort_cov_dir} " \
                  f"--output-file {cov_dir}/sp_ORT_UT2.info " \
                  f"--rc lcov_branch_coverage=1"
    subprocess.run(cov_collect, shell=True)


def single_task(new_ir_file_path):
    print(new_ir_file_path)
    ret_code, _, stderr = run_test(new_ir_file_path)
    if ret_code:  # detected a bug
        # if "NOT_IMPLEMENTED" in stderr:
        #     print(f"[WARNING] SKip the unsupported issue...")
        #     return
        if "Unable to handle object of type" in stderr:
            print(f"[WARNING] SKip the FP arising from invalid seed graph...")
            return
        stderr = fuzz_utils.simple_crash_message(stderr)
        unique_bug_mess = fuzz_utils.extract_unique_crash_message(stderr)
    print("yes!")


class Fuzzer:
    def __init__(self,
                 donor_dir,
                 base_irs_dir,
                 execution_time,
                 log_file,
                 failure_dir,
                 execution_dir,
                 fuzz_mod="random"):
        self.DonorPool = irs_utils.IRsPool(donor_dir, max_num=100)
        self.BaseIRsPool = irs_utils.IRsPool(base_irs_dir, max_num=4000)
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


    def fuzz(self):
        total_mutant_num = 0

        all_tasks_list = []
        pool = multiprocessing.Pool(processes=20)

        for k, v in self.BaseIRsPool.irs_pool.items():
            for base_seed_clazz in v:
                total_mutant_num += 1
                if total_mutant_num > 4000:
                    break
                tmp_path = f"{execution_dir}/{total_mutant_num}.onnx"
                shutil.copy(base_seed_clazz.filename, tmp_path)
                all_tasks_list.append(tmp_path)
        pool.map(single_task, all_tasks_list)
        pool.close()
        pool.join()
        print("Finish All")
        collect_cov(cov_dir=f"../res/cov_cumulate/ORT/ablation")  # coverage saved here!


if __name__ == "__main__":
    """
    Run all seed  or all UTs: (plz change the variable path:)
    1. change the "base_irs_dir_nnsmith" (Line-119), 
    2. "res_dir" (Line-125), and 
    3. "output_cov_file_name" (Line-35)
    """
    res_dir = '../res'
    donor_dir = sys.argv[1] if len(sys.argv) > 1 else f"{res_dir}/onnx_ut/onnx_models_gf"
    # base_irs_dir_nnsmith = sys.argv[2] if len(sys.argv) > 2 else f"{res_dir}/onnx_nnsmith"
    base_irs_dir_nnsmith = sys.argv[2] if len(sys.argv) > 2 else f"{res_dir}/onnx_ut/onnx_models_gf"
    base_irs_dir = [base_irs_dir_nnsmith]

    execution_time = sys.argv[3] if len(sys.argv) > 3 else "12h"
    res_dir = f"{res_dir}/CCS25/ORT/Seed_NNSmith4k"
    res_dir = f"{res_dir}/CCS25/ORT/all_UT"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
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
