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
    plt.xlabel('Time (hours)',fontweight='bold')
    plt.ylabel(f'#Coverage (1000 {granularity})', fontweight='bold')
    plt.xticks(ticks, fontsize=22, fontweight='bold')
    plt.yticks(fontsize=22, fontweight='bold')

    plt.tight_layout()
    plt.legend(loc='lower right', fontsize=20)
    plt.savefig(f"trends_{granularity}.pdf")
    plt.show()


def preprocess(cov_data):
    cov_data = {k: v[:24] for k, v in cov_data.items()}
    for k, cov_list in cov_data.items():
        # if k == 'WhiteFox':
        #     cov_list = double_data(cov_list)
        new_cov_list = [i/1000 for i in cov_list]
        new_cov_list.insert(0, 0)
        print(len(new_cov_list))
        cov_data[k] = new_cov_list

    return cov_data


def plot_coverage_smooth(data, granularity):
    time_points = np.linspace(0, 12, num=25)  # 原始时间点
    ticks = range(0, 13, 1)

    plt.figure(figsize=(10, 7))
    styles = {
        'HirGen': ('-o', '#1f99b4'),  # 蓝色
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
    plt.ylabel(f'#Coverage (1k {granularity})',  fontsize=26)
    plt.xticks(ticks, fontsize=26)
    plt.yticks(fontsize=26)

    plt.tight_layout()
    plt.legend(loc='lower right', fontsize=26)
    plt.savefig(f"trends_{granularity}.pdf")
    plt.show()


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
        'NNSmith': [16955, 16995, 16995, 17030, 17040, 17040, 17055, 17055, 17055, 17056, 17068, 17068, 17068, 17068,
                    17068, 17068, 17068, 17068, 17068, 17068, 17069, 17069, 17069, 17069, 17069],
        # 'HirGen': [15068, 15068, 15068, 15068, 15069, 15069, 15069, 15069, 15069, 16172, 16172, 16172, 16172, 16172,
        #            16172, 16172, 16172, 16172, 16172, 16172, 16172, 16172, 16172, 16172, 16172],
        'WhiteFox': [11800, 15303, 15503, 15503, 15567, 15583, 15652, 15657, 15657, 15657, 15660, 15994, 15994, 16036,
                     16087, 16087, 16087, 16210, 16256, 16256, 16265, 16306, 16306, 16306],
        'OATest': [26913, 27395, 27509, 27622, 27651, 27731, 27798, 27911, 27942, 28014, 28032, 28041, 28070,
                        28140, 28145, 28145, 28153, 28221, 28221, 28259, 28266, 28270, 28270, 28282, 28283]}
    func_cov_dict = {
        'NNSmith': [15539, 15583, 15584, 15596, 15599, 15603, 15604, 15604, 15604, 15604, 15609, 15610, 15610, 15610,
                    15610, 15610, 15610, 15610, 15610, 15610, 15610, 15610, 15610, 15610, 15610],
        # 'HirGen': [14860, 14860, 14860, 14860, 14862, 14862, 14862, 14862, 14862, 15301, 15301, 15301, 15301, 15301,
        #            15301, 15301, 15301, 15301, 15301, 15301, 15301, 15301, 15301, 15301, 15301],
        'WhiteFox': [13918, 15237, 15299, 15299, 15314, 15319, 15329, 15332, 15332, 15332, 15332, 15436, 15436, 15443,
                     15473, 15473, 15473, 15518, 15546, 15546, 15554, 15558, 15558, 15558],
        'OATest': [18894, 19060, 19109, 19155, 19164, 19192, 19208, 19292, 19307, 19317, 19323, 19346, 19360,
                        19371, 19378, 19378, 19382, 19394, 19394, 19397, 19399, 19399, 19401, 19405, 19406]}
    branch_cov_dict = {
        'NNSmith': [36613, 36814, 36832, 36990, 37012, 37077, 37107, 37107, 37107, 37108, 37194, 37220, 37220, 37220,
                    37220, 37220, 37220, 37221, 37221, 37221, 37223, 37223, 37223, 37223, 37223],
        # 'HirGen': [30665, 30665, 30665, 30665, 30669, 30669, 30669, 30669, 30669, 32993, 32993, 32993, 32993, 32993,
        #            32993, 32993, 32993, 32993, 32993, 32993, 32993, 32993, 32993, 32993, 32993],
        'WhiteFox': [24558, 33955, 34617, 34617, 34798, 34833, 35038, 35069, 35069, 35069, 35089, 35606, 35606, 35707,
                     36065, 36065, 36065, 36494, 36642, 36643, 36643, 36746, 36746, 36746],
        'OATest': [58314, 59852, 60558, 60995, 61113, 61344, 61543, 61884, 62042, 62238, 62272, 62558, 62642,
                        62842, 62857, 62858, 62922, 63128, 63131, 63261, 63304, 63337, 63378, 63436, 63460]}

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
    #
    plot_coverage_smooth(branch_cov_dict, "branches")
    plot_coverage_smooth(line_cov_dict, "lines")
    # plot_coverage_smooth(func_cov_dict, "functions")

