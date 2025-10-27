#!/usr/bin/env python3
import sys
from collections import defaultdict


def merge_brda_records(info_file, output_file):
    """
    读取 .info 文件，将 BRDA 记录按 (line, block) 分组后进行合并：
      - 如果该组内所有记录均为 "-"，则合并后 hit 仍为 "-"。
      - 如果分组内记录数为2（简单条件），则合并后 hit 为两条记录 hit 数之和。
      - 如果分组内记录数大于2（复合条件），则只要任一记录 hit 数大于0，则合并后 hit 记为1；否则记为0。
      - 对于其他情况，作为 fallback 采用求和。
    """
    groups = defaultdict(list)  # key: (line, block)，value: list of hit 字符串

    with open(info_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("BRDA:"):
                # 假设记录格式为：BRDA:<line>,<block>,<branch>,<hit>
                parts = line[5:].split(',')
                if len(parts) != 4:
                    continue
                line_no, block, branch, hit = parts
                key = (line_no, block)
                groups[key].append(hit)
            else:
                continue
                # 非 BRDA 行直接输出
                # print(line)

    merged_records = []
    total_hit_count = 0  # 用来累计所有记录的 hit 次数
    total_branch_count = 0  # 用来累计所有的分支数

    for key, hits in groups.items():
        count = len(hits)
        if all(h == '-' for h in hits):
            merged_hit = '-'
        else:
            if count == 2:
                # 简单条件：对两个分支求和
                merged_hit = sum(int(h) if h != '-' else 0 for h in hits)
            elif count > 2:
                # 复合条件：只要任一分支被执行，则认为 if 条件执行（计为1）
                merged_hit = 1 if any(h != '-' and int(h) > 0 for h in hits) else 0
            else:
                # fallback：直接求和
                merged_hit = sum(int(h) if h != '-' else 0 for h in hits)

        # 累计当前组的 hit 数
        group_hit_count = merged_hit if merged_hit != '-' else 0
        total_hit_count += group_hit_count
        total_branch_count += 2
        # 构造合并后的 BRDA 记录；这里保留 line 和 block，branch 用 "*" 表示合并
        merged_record = f"BRDA:{key[0]},{key[1]},*,{merged_hit}"
        merged_records.append(merged_record)

    # 计算分支覆盖率
    branch_coverage = (total_hit_count / total_branch_count) * 100 if total_branch_count > 0 else 0

    # # 写入合并后的记录到指定的输出文件
    # with open(output_file, 'w') as f_out:
    #     for record in merged_records:
    #         f_out.write(record + '\n')
    # print(f"Merged .info file written to {output_file}")
    print(f"Total hit count: {total_hit_count}")
    print(f"Total branch count: {total_branch_count}")
    print(f"Branch coverage: {branch_coverage:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: {} <coverage.info> <output.info>".format(sys.argv[0]))
        sys.exit(1)

    info_file = sys.argv[1]
    output_file = sys.argv[2]

    merge_brda_records(info_file, output_file)
