import os
import re
import Levenshtein


def parse_execution_time(execution_time):
    time_unit = execution_time[-1]
    time_value = int(execution_time[:-1])
    if time_unit == 's':
        return time_value
    elif time_unit == 'm':
        return time_value * 60
    elif time_unit == 'h':
        return time_value * 3600
    else:
        raise ValueError("Invalid time unit. Use 's' for seconds, 'm' for minutes, or 'h' for hours.")


def save_test_case(file_dir, filename, content):
    filepath = os.path.join(file_dir, filename)
    with open(filepath, 'w') as f:
        f.write(content)
    return filepath


def simple_crash_message(message):
    stacktrace_flag = 'Traceback (most recent call last):\n'
    if stacktrace_flag in message:
        simple_res = message.split(stacktrace_flag)[-1]
    else:
        simple_res = 'Segmentation fault (core dumped)'
    simple_res = simple_res.split('# from tvm.script import ')[0]
    return simple_res


def extract_unique_crash_message(stack_trace):
    stack_trace = stack_trace.strip()
    stack_trace = stack_trace.split("\n")[-1]
    if not stack_trace.startswith("Op("):
        stack_trace = re.sub(r"\[.*\]|\(.*\)", "", stack_trace)
    stack_trace = re.sub(r'\d+', "**", stack_trace)
    stack_trace = re.sub(r" .*?\..*?(_handle)", r' XX.\1', stack_trace)
    return stack_trace.strip()


def is_duplicate_bug(this_bug, bug_set):
    def _is_same_bug(s1, s2, threshold=.8):
        if s1.split(":")[0] != s2.split(":")[0]:  # TypeError vs ValueError
            return False
        distance = Levenshtein.distance(s1, s2)
        max_len = max(len(s1), len(s2))
        sim_score = 1 - (distance/max_len)
        return sim_score >= threshold

    for bug in bug_set:
        if _is_same_bug(this_bug, bug):
            return True
    return False


def get_all_unique_bugs(bug_file):
    all_bugs = set()
    with open(bug_file) as f:
        bugs = f.read()
    for item in bugs.split("==================================================================")[:-1]:
        res = extract_unique_crash_message(item)
        if res not in all_bugs and not is_duplicate_bug(res, all_bugs):
                all_bugs.add(res)
                print(res)
                # print('*'*100)
    print(len(all_bugs))
    return all_bugs


if __name__ == '__main__':
    ut_pure = get_all_unique_bugs("../res/res_fuzzer_log.txt")
    # ut_ut = get_all_unique_bugs("../res/res_ut_ut.txt")
    # ut_nnsmith = get_all_unique_bugs("../res/res_ut_nnsmith.txt")
    # ut_hirgen = get_all_unique_bugs("../res/res_ut_hirgen.txt")
    # total_bugs = ut_hirgen - ut_nnsmith - ut_ut
    # print(total_bugs)
    # print(len(total_bugs))
