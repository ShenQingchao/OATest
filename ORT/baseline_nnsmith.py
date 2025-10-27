import os
import sys
import time
from datetime import datetime
import shutil
import subprocess
import onnx
import fuzz_utils


def run_test(model_path):
    process = subprocess.Popen(
        ["python", "test_ort.py", model_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode(), stderr.decode()


def collect_cov(cov_cnt, cov_dir="."):
    ort_cov_dir = "/software/onnxruntime/build/Linux/RelWithDebInfo/CMakeFiles/onnxruntime_optimizer.dir/software/onnxruntime/onnxruntime/"
    cov_collect = f"lcov --capture --directory {ort_cov_dir} " \
                  f"--output-file {cov_dir}/sp_ORT_NNSmith4k_{cov_cnt}.info " \
                  f"--rc lcov_branch_coverage=1"
    subprocess.run(cov_collect, shell=True)


class Fuzzer:
    def __init__(self,
                 base_irs_dir,
                 execution_time,
                 log_file,):
        self.base_ir_dir = base_irs_dir
        self.log_file = log_file
        self.start_time = time.time()
        self.execution_time = fuzz_utils.parse_execution_time(execution_time)

        self.all_detected_unique_bugs = []
        self.invalid_test_num = 0

    def log_bug(self, test_case_path, stderr):
        with open(self.log_file, 'a') as f:
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Test Case Path: {test_case_path}\n")
            f.write(f"ERROR: {stderr}\n")
            f.write("=" * 66 + "\n")
        failure_dir = os.path.join(self.base_ir_dir[0], "failed_tests")
        shutil.copy(test_case_path, failure_dir)

    def single_task(self, new_ir_file_path):
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
            if unique_bug_mess not in self.all_detected_unique_bugs or "Segmentation fault" in unique_bug_mess:
                print(unique_bug_mess)
                self.all_detected_unique_bugs.append(unique_bug_mess)
                self.log_bug(new_ir_file_path, stderr)
                print(f"[INFO] Detected bug number: {len(self.all_detected_unique_bugs)}")
        else:
            print(f"[INFO] {new_ir_file_path} successfully!")

    def fuzz(self):
        total_test_num = 0

        def find_onnx_files(root_dir):
            onnx_files = []
            for dirpath, _, filenames in os.walk(root_dir):
                for filename in filenames:
                    if filename.endswith('.onnx'):
                        onnx_files.append(os.path.join(dirpath, filename))
            return onnx_files

        all_models_list = find_onnx_files(self.base_ir_dir[0])

        last_save_cov_time = time.time()
        cov_during_time = 900
        cov_cnt = 0

        tmp_dir = f"./tmp"
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)

        while True:
            for model_path in all_models_list:
                current_time = time.time()
                during_time = current_time - self.start_time

                if during_time > self.execution_time:
                    collect_cov(cov_cnt)
                    print(f"[INFO]: Total generated tests number is: {total_test_num};"
                          f"Invalid tests number is:{self.invalid_test_num}; "
                          f"Valid rate is: {1 - self.invalid_test_num / total_test_num}")
                    print("[INFO]: Finished ALL && Timeout!")
                    shutil.rmtree("./tmp")
                    return True
                print(model_path)
                model = onnx.load(model_path)
                total_test_num += 1
                if not onnx.checker.check_model(model, full_check=True):
                    self.invalid_test_num += 1

                new_path = os.path.join(tmp_dir, os.path.split(model_path)[-1])
                shutil.copy(model_path, new_path)
                self.single_task(new_path)

                current_time = time.time()
                during_time = current_time - last_save_cov_time
                if during_time > cov_during_time:
                    collect_cov(cov_cnt)
                    cov_cnt += 1
                    last_save_cov_time = time.time()
            print("[INFO] Finish generating a mutation for each pass group!")

            if 'onnx_ut' in self.base_ir_dir[0]:
                collect_cov(cov_cnt)
                print(f"[INFO]: Total generated tests number is: {total_test_num};"
                      f"Invalid tests number is:{self.invalid_test_num}; "
                      f"Valid rate is: {1 - self.invalid_test_num / total_test_num}")
                print("[INFO]: Finished ALL && Timeout!")
                shutil.rmtree("./tmp")
                return True


if __name__ == "__main__":
    res_dir = '../res'
    # base_irs_dir = sys.argv[1] if len(sys.argv) > 1 else f"{res_dir}/onnx_ut/onnx_models_gf"
    base_irs_dir = sys.argv[1] if len(sys.argv) > 1 else f"{res_dir}/onnx_nnsmith"
    base_irs_dir = [base_irs_dir]

    execution_time = sys.argv[2] if len(sys.argv) > 2 else "12h"
    res_dir = os.path.join(res_dir, "onnx_res_ut_1229")
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    log_file = sys.argv[3] if len(sys.argv) > 3 else f"{res_dir}/res_fuzzer_log.txt"

    fuzzer = Fuzzer(base_irs_dir=base_irs_dir,
                    execution_time=execution_time,
                    log_file=log_file)
    fuzzer.fuzz()
