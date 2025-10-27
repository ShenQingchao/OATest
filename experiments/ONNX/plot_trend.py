import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


def plot_coverage(data, granularity):
    # time_points = list(range(1, 21))
    time_points = np.linspace(0, 12, num=25)
    ticks = range(0, 13, 1)
    # print(time_points)
    # print(list(ticks))

    plt.figure(figsize=(10, 7))
    styles = {
        'HirGen':  ('-o', '#1f99b4'),  # 蓝色
        'NNSmith': ('--s', '#ff7f0e'),  # 橙色
        'WhiteFox': (':x', '#9467bd'),    # 红色
        'OATest': ('-.v', '#2ca02c'),  # 绿色
    }

    for method, coverage in data.items():
        line_style, color = styles.get(method, ('-', 'black'))
        plt.plot(time_points, coverage, line_style, color=color, label=method,
                 markersize=9, linewidth=3)

    # plt.title('Coverage Over Time')
    plt.xlabel('Time (hours)', fontweight='bold')
    plt.ylabel(f'#Coverage (1000 {granularity})', fontweight='bold')
    plt.xticks(ticks, fontsize=22, fontweight='bold')
    plt.yticks(fontsize=22, fontweight='bold')

    plt.tight_layout()
    plt.legend(loc='lower right', fontsize=20)
    plt.savefig(f"trends_ort_{granularity}.pdf")
    plt.show()


def plot_coverage_smooth(data, granularity):
    time_points = np.linspace(0, 12, num=25)  # 原始时间点
    ticks = range(0, 13, 1)

    plt.figure(figsize=(10, 7))
    styles = {
        'Seed Tests': ('-o', '#1f99b4'),  # 蓝色
        'NNSmith': ('--s', '#ff7f0e'),  # 橙色
        'WhiteFox': (':x', '#9467bd'),  # 紫色
        'OATest': ('-.v', '#2ca02c'),  # 绿色
    }

    for method, coverage in data.items():
        line_style, color = styles.get(method, ('-', 'black'))  # 获取样式和颜色

        # 使用PCHIP插值，生成平滑曲线
        pchip = PchipInterpolator(time_points, coverage)
        smooth_time = np.linspace(time_points.min(), time_points.max(), 300)  # 插值密度更高
        smooth_coverage = pchip(smooth_time)

        # 绘制平滑曲线（不标记插值点），并添加图例
        line, = plt.plot(smooth_time, smooth_coverage, line_style[:-1], color=color, linewidth=3, label=method)

        # 在原始数据点标记样式
        plt.plot(time_points, coverage, line_style[-1], color=color, linestyle='None', markersize=9)

    # 设置图例
    plt.xlabel('Time (hours)', fontsize=26)
    plt.ylabel(f'#Coverage (1k {granularity})', fontsize=26)
    plt.xticks(ticks, fontsize=26)
    plt.yticks(fontsize=26)

    plt.tight_layout()
    plt.legend(loc='lower right', fontsize=26)
    plt.savefig(f"trends_ort_{granularity}.pdf")
    plt.show()


def preprocess(cov_data):
    cov_data = {k: v[:24] for k, v in cov_data.items()}
    for k, cov_list in cov_data.items():
        # if k == 'WhiteFox':
        #     cov_list = double_data(cov_list)
        new_cov_list = [i/1000 for i in cov_list]
        new_cov_list.insert(0, 0)
        # print(len(new_cov_list))
        cov_data[k] = new_cov_list

    return cov_data


if __name__ == '__main__':
    config = {
        "font.family": "sans-serif",  # 使用衬线体
        "font.sans-serif": ["Helvetica"],  # 全局默认使用衬线宋体,
        "font.size": 22,
        "axes.unicode_minus": False,
        # "mathtext.fontset": "stix",  # 设置 LaTeX 字体，stix 近似于 Times 字体
    }
    matplotlib.rcParams['xtick.labelsize'] = 15
    plt.rcParams.update(config)

    line_cov_dict = {
        'NNSmith': [3028, 3382, 3431, 3500, 3523, 3732, 3777, 3788, 3802, 3811, 3818, 3822, 3827, 3827, 3886, 3889,
                    4017, 4020, 4042, 4042, 4043, 4043, 4043, 4057],
        'WhiteFox': [1349, 1500, 1517, 1643, 1757, 2062, 2187, 2196, 2209, 2572, 2572, 2572, 2843, 2843, 2877, 3037,
                     3040, 3179, 3179, 3179, 3179, 3179, 3179, 3179],
        'OATest': [5112, 5185, 5424, 5856, 5904, 5910, 5913, 6177, 6338, 6427, 6427, 6427, 6427, 6427, 6427, 6441,
                        6510, 6515, 6591, 6591, 6591, 6591, 6591, 6591],
        }  # 'Seed Tests': [6050 for i in range(25)]
    func_cov_dict = {
        'NNSmith': [420, 465, 466, 472, 473, 489, 490, 490, 490, 491, 491, 491, 491, 491, 492, 492, 498, 499, 499, 499,
                    499, 499, 499, 500],
        'WhiteFox': [251, 266, 267, 277, 285, 316, 331, 332, 334, 372, 372, 372, 398, 398, 398, 423, 423, 430, 430, 430,
                     430, 430, 430, 430],
        'OATest': [515, 522, 532, 554, 559, 559, 560, 574, 581, 584, 584, 584, 584, 584, 584, 585, 585, 585, 596,
                        596, 596, 596, 596, 596]}
    branch_cov_dict = {
        'NNSmith': [3328, 3676, 3747, 3826, 3861, 4070, 4140, 4163, 4185, 4203, 4215, 4235, 4251, 4251, 4322, 4332,
                    4484, 4488, 4542, 4543, 4545, 4545, 4546, 4564],
        'WhiteFox': [1388, 1567, 1603, 1786, 1972, 2344, 2453, 2464, 2474, 2847, 2847, 2848, 3126, 3127, 3173, 3311,
                     3322, 3572, 3572, 3572, 3572, 3572, 3572, 3572],
        'OATest': [5685, 5777, 6060, 6653, 6708, 6738, 6757, 7107, 7322, 7456, 7456, 7456, 7456, 7456, 7457, 7471,
                        7566, 7579, 7729, 7732, 7732, 7732, 7732, 7732],
        # 'Seed Tests': [6878 for i in range(25)]
        }

    branch_cov_dict = preprocess(branch_cov_dict)
    line_cov_dict = preprocess(line_cov_dict)
    func_cov_dict = preprocess(func_cov_dict)
    print("Branch cover:")
    for k, v in branch_cov_dict.items():
        print(k, v[-1]*1000)
    print()
    print("Line cover:")
    for k, v in line_cov_dict.items():
        print(k, v[-1]*1000)
    print()
    print("Func cover:")
    for k, v in func_cov_dict.items():
        print(k, v[-1]*1000)

    # plot_coverage(branch_cov_dict, "branches")
    # plot_coverage(line_cov_dict, "lines")
    # plot_coverage(func_cov_dict, "functions")

    plot_coverage_smooth(branch_cov_dict, "branches")
    plot_coverage_smooth(line_cov_dict, "lines")
    # plot_coverage_smooth(func_cov_dict, "functions")


