import re
import os


def parse_info_file(filename, target_folder):
    with open(filename, 'r') as file:
        lines = file.readlines()

    coverage_data = {'lines': set(), 'functions': set(), 'branches': set()}
    source_total_num = {'lines': 0, 'functions': 0, 'branches': 0}

    current_file = None
    for line in lines:
        if line.startswith('SF:'):
            current_file = line.strip().split(':')[1]

        elif current_file and target_folder in current_file:

            # current_file = current_file.split('/data/shenqingchao/zibo')[-1]
            # print(current_file)
            # 'support','driver','target','auto_scheduler', 'ir',
            if current_file.split("/")[4] in ['relay', 'te', 'contrib', 'autotvm', 'runtime', 'topi']:
               continue
            if line.startswith('FNDA:'):
                parts = line.strip().split(':')
                execution_count = int(parts[1].split(',')[0])
                function_name = parts[1].split(',')[1]
                source_total_num['functions'] += 1
                if execution_count > 0:
                    coverage_data['functions'].add((current_file, function_name))
            elif line.startswith('DA:'):
                parts = line.strip().split(':')
                line_info = parts[1].split(',')
                line_number = int(line_info[0])
                execution_count = int(line_info[1])
                source_total_num['lines'] += 1
                if execution_count > 0:
                    coverage_data['lines'].add((current_file, line_number))
            elif line.startswith('BRDA:'):
                parts = line.strip().split(':')
                branch_info = parts[1].split(',')
                line_number = int(branch_info[0])
                block_number = int(branch_info[1])
                branch_number = int(branch_info[2])
                taken = branch_info[3]
                source_total_num['branches'] += 1
                if taken != '0' and taken != '-':
                    coverage_data['branches'].add((current_file, line_number, block_number, branch_number))

    print(f"Total coverage of {filename}:")
    print(f"Lines: {len(coverage_data['lines'])} / {source_total_num['lines']} = {len(coverage_data['lines'])/source_total_num['lines']}")
    print(f"Functions:{len(coverage_data['functions'])} / {source_total_num['functions']} = {len(coverage_data['functions'])/source_total_num['functions']}")
    print(f"Branches:{len(coverage_data['branches'])} / {source_total_num['branches']} = {len(coverage_data['branches'])/source_total_num['branches']}")
    print('\n\n')

    res_file_base = f"cov_{os.path.split(filename)[-1].split('.')[0]}"
    for granularity in ['lines', 'functions', 'branches']:
        res_file = f"{res_file_base}_{granularity}.txt"
        with open(res_file, 'w') as f:
            for item in coverage_data[granularity]:
                f.write(f"{item}\n")
    return coverage_data, source_total_num


def unique_coverage(file1_data, file2_data):
    unique_to_file1 = {
        'lines': file1_data['lines'] - file2_data['lines'],
        'functions': file1_data['functions'] - file2_data['functions'],
        'branches': file1_data['branches'] - file2_data['branches']
    }

    unique_to_file2 = {
        'lines': file2_data['lines'] - file1_data['lines'],
        'functions': file2_data['functions'] - file1_data['functions'],
        'branches': file2_data['branches'] - file1_data['branches']
    }
    return unique_to_file1, unique_to_file2


def combine_coverage(file1_data, file2_data):
    combined_cov_dict = {
        'lines': file1_data['lines'].union(file2_data['lines']),
        'functions': file1_data['functions'].union(file2_data['functions']),
        'branches': file1_data['branches'].union(file2_data['branches'])
    }
    return combined_cov_dict


def get_sub_unique_cov_bran(unique_cov_dict):
    group_cov = {}
    # print(unique_to_coverage2['lines'])
    for k, v,v2,v3 in unique_cov_dict:
        sub_category = "/".join(k.split('/')[:5])
        if sub_category not in group_cov.keys():
            group_cov[sub_category] = []
        group_cov[sub_category].append(v)

    for k, v in group_cov.items():
        print(k, len(v))
    print('*'*20)


def get_sub_unique_cov(unique_cov_dict):
    group_cov = {}
    # print(unique_to_coverage2['lines'])
    for k, v in unique_cov_dict:
        sub_category = "/".join(k.split('/')[:5])
        if sub_category not in group_cov.keys():
            group_cov[sub_category] = []
        group_cov[sub_category].append(v)

    for k, v in group_cov.items():
        print(k, len(v))
    print('*'*20)


def get_all_unique_cov(target_folder, seed_cov_file_list, our_cov_file_list):
    # total coverage
    coverage1_data = None
    coverage2_data = None
    source_total_num = None
    for cov_info in seed_cov_file_list:
        this_cov_data, source_total_num = parse_info_file(cov_info, target_folder)
        if coverage1_data:
            coverage1_data = combine_coverage(coverage1_data, this_cov_data)
        else:
            coverage1_data = this_cov_data

    for cov_info in our_cov_file_list:
        this_cov_data, _ = parse_info_file(cov_info, target_folder)
        if coverage2_data:
            coverage2_data = combine_coverage(coverage2_data, this_cov_data)
        else:
            coverage2_data = this_cov_data

    # unique coverage
    unique_to_coverage1, unique_to_coverage2 = unique_coverage(coverage1_data, coverage2_data)
    print("\nUnique to first method")
    print("Lines:", len(unique_to_coverage1['lines']))
    print("Functions:", len(unique_to_coverage1['functions']))
    print("Branches:", len(unique_to_coverage1['branches']))

    print("\nUnique to second method:")
    print("Lines:", len(unique_to_coverage2['lines']))
    print("Functions:", len(unique_to_coverage2['functions']))
    print(f"Branches: {len(unique_to_coverage2['branches'])} ; {len(unique_to_coverage2['branches'])/source_total_num['branches']}")
    print()

    print("unique coverage of second coverage about sub category")
    # get_sub_unique_cov(unique_to_coverage2['lines'])
    # get_sub_unique_cov(unique_to_coverage2['functions'])
    get_sub_unique_cov_bran(unique_to_coverage2['branches'])


if __name__ == '__main__':
    target_folder = '/software/tvm/src'
    target_folder = '/tvm/src'

    # target_folder = '/software/tvm/src/relax/transform'
    # target_folder = '/software/tvm/src/tir'

    seed_cov_file_list = []
    optfuzz_cov_file_list = []

    base_dir = "/software/tvm/_cov"

    cov_ut = f'{base_dir}/sp_ut.info'
    cov_nnsmith = f'{base_dir}/cov_nnsmith12.info'
    cov_hirgen = f'{base_dir}/cov_hirgen12.info'
    # cov_whitefox = f'{base_dir}/cov_whitefox.info'
    # seed_cov_file_list.extend([cov_ut, cov_nnsmith, cov_hirgen, cov_whitefox])
    seed_cov_file_list.extend([cov_ut, cov_nnsmith, cov_hirgen])

    # cov_optfuzz_ut = f'{base_dir}/cov_optfuzz_block.info'
    # cov_optfuzz_nnsmith = f'{base_dir}/cov_optfuzz_nnsmith12.info'
    # cov_optfuzz_hirgen = f'{base_dir}/cov_optfuzz_hirgen.info'
    # cov_modeltailor_nnsmith = f'{base_dir}/cov_modeltailor_nnsmith.info'
    cov_modeltailor_nnsmith = f"{base_dir}/sp_MT_nnsmith.info"
    cov_modeltailor_ut = f"{base_dir}/sp_MT_ut2k.info"

    cov_modeltailor_hirgen = f"{base_dir}/sp_MT_hirgen.info"

    # optfuzz_cov_file_list.extend([cov_optfuzz_ut, cov_optfuzz_nnsmith, cov_optfuzz_hirgen, cov_modeltailor_nnsmith])
    # optfuzz_cov_file_list.extend([cov_modeltailor_nnsmith, cov_modeltailor_ut, cov_modeltailor_hirgen])

    get_all_unique_cov(target_folder, seed_cov_file_list, optfuzz_cov_file_list)

    # print("Use different sources!")
    get_all_unique_cov(target_folder, [cov_modeltailor_ut], [cov_modeltailor_hirgen, cov_modeltailor_nnsmith])
    # get_all_unique_cov(target_folder, [cov_modeltailor_ut], [cov_modeltailor_nnsmith])
    # get_all_unique_cov(target_folder, [cov_modeltailor_nnsmith], [cov_modeltailor_hirgen])

    print("Donor vs Seed vs synthesis:")
    cov_ut = f'{base_dir}/sp_ut.info'
    cov_nnsmith2k = f'{base_dir}/sp_nnsmith2k.info'
    cov_hirgen2k = f"{base_dir}/sp_hirgen2k.info"
    get_all_unique_cov(target_folder, [cov_ut], [cov_nnsmith2k, cov_hirgen2k])
