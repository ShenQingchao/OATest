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
    plt.ylabel(f'{granularity} Coverage', fontweight='bold')
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
    plt.ylabel(f'{granularity} Coverage (%)', fontsize=26)
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
        # new_cov_list = [i for i in cov_list]
        new_cov_list = cov_list
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
        'NNSmith': [31.271300216874938, 34.927191985954764, 35.43323350201384, 36.14582257564805, 36.38335226685944,
                    38.54177424351957, 39.006506248063616, 39.120107404729936, 39.264690695032535, 39.35763709594134,
                    39.42992874109264, 39.47123825260766, 39.522875142001446, 39.522875142001446, 40.132190436848084,
                    40.163172570484356, 41.4850769389652, 41.51605907260147, 41.74326138593411, 41.74326138593411,
                    41.753588763812864, 41.753588763812864, 41.753588763812864, 41.89817205411546],
        'WhiteFox': [13.79204580308762, 15.335855229526633, 15.509661588794602, 16.797873428074837, 17.963398425518864,
                     21.081688988855944, 22.359676924649833, 22.451692056026992, 22.584602801349554, 26.295879766895002,
                     26.295879766895002, 26.295879766895002, 29.066557611696147, 29.066557611696147, 29.414170330232082,
                     31.049994888048253, 31.08066659850731, 32.50178918311011, 32.50178918311011, 32.50178918311011,
                     32.50178918311011, 32.50178918311011, 32.50178918311011, 32.50178918311011],
        'OATest': [43.182970096300046, 43.79962831559385, 45.81855043081602, 49.467815509376585,
                        49.873289406994424, 49.92397364419666, 49.94931576279777, 52.17942219969589, 53.53944923128907,
                        54.29126541645548, 54.29126541645548, 54.29126541645548, 54.29126541645548, 54.29126541645548,
                        54.29126541645548, 54.40952863659402, 54.992397364419666, 55.034634228754854,
                        55.676634566649774, 55.676634566649774, 55.676634566649774, 55.676634566649774,
                        55.676634566649774, 55.676634566649774]}
    func_cov_dict = {
        'NNSmith': [48.78048780487805, 54.00696864111498, 54.12311265969802, 54.81997677119629, 54.93612078977933,
                    56.79442508710801, 56.91056910569105, 56.91056910569105, 56.91056910569105, 57.026713124274096,
                    57.026713124274096, 57.026713124274096, 57.026713124274096, 57.026713124274096, 57.14285714285714,
                    57.14285714285714, 57.8397212543554, 57.955865272938446, 57.955865272938446, 57.955865272938446,
                    57.955865272938446, 57.955865272938446, 57.955865272938446, 58.07200929152149],
        'WhiteFox': [29.050925925925924, 30.787037037037035, 30.90277777777778, 32.06018518518518, 32.98611111111111,
                     36.574074074074076, 38.31018518518518, 38.425925925925924, 38.657407407407405, 43.05555555555556,
                     43.05555555555556, 43.05555555555556, 46.06481481481482, 46.06481481481482, 46.06481481481482,
                     48.95833333333333, 48.95833333333333, 49.76851851851852, 49.76851851851852, 49.76851851851852,
                     49.76851851851852, 49.76851851851852, 49.76851851851852, 49.76851851851852],
        'OATest': [59.814169570267126, 60.62717770034843, 61.78861788617886, 64.34378629500581, 64.92450638792103,
                        64.92450638792103, 65.04065040650406, 66.66666666666666, 67.47967479674797, 67.8281068524971,
                        67.8281068524971, 67.8281068524971, 67.8281068524971, 67.8281068524971, 67.8281068524971,
                        67.94425087108013, 67.94425087108013, 67.94425087108013, 69.22183507549362, 69.22183507549362,
                        69.22183507549362, 69.22183507549362, 69.22183507549362, 69.22183507549362]}
    branch_cov_dict = {
        'NNSmith': [29.658048373644704, 33.07756463719767, 33.76146788990826, 34.612176814011676, 34.862385321100916,
                    37.164303586321935, 37.69808173477898, 37.79816513761468, 37.998331943286075, 38.09841534612177,
                    38.14845704753962, 38.21517931609675, 38.298582151793156, 38.298582151793156, 38.982485404503755,
                    39.0325271059216, 40.550458715596335, 40.56713928273562, 40.783986655546286, 40.783986655546286,
                    40.80066722268557, 40.80066722268557, 40.80066722268557, 41.03419516263553],
        'WhiteFox': [12.726972598217234, 14.26213271706834, 14.4106965995378, 15.764278639815121, 17.1343677781446,
                     20.287223506107626, 21.360184879498185, 21.426213271706835, 21.54176295807197, 25.27236711786068,
                     25.27236711786068, 25.27236711786068, 28.029052492571804, 28.029052492571804, 28.39220864971938,
                     29.861340376361834, 29.910861670518322, 31.561571475734567, 31.561571475734567, 31.561571475734567,
                     31.561571475734567, 31.561571475734567, 31.561571475734567, 31.561571475734567],
        'OATest': [41.02850754611515, 41.57350475125769, 43.278367803242034, 47.27501397428731, 47.764114030184466,
                        47.82001117942985, 47.847959754052546, 50.26551145891559, 51.858580212409166, 52.6690888764673,
                        52.6690888764673, 52.6690888764673, 52.6690888764673, 52.6690888764673, 52.6690888764673,
                        52.73896031302403, 53.46562325321409, 53.47959754052544, 54.27613191727222, 54.27613191727222,
                        54.27613191727222, 54.27613191727222, 54.27613191727222, 54.27613191727222]}

    branch_cov_dict = preprocess(branch_cov_dict)
    line_cov_dict = preprocess(line_cov_dict)
    func_cov_dict = preprocess(func_cov_dict)
    print("Branch cover:")
    for k, v in branch_cov_dict.items():
        print(k, round(v[-1] / 100, 4))
    print()
    print("Line cover:")
    for k, v in line_cov_dict.items():
        print(k, round(v[-1] / 100, 4))
    # print()
    # print("Func cover:")
    # for k, v in func_cov_dict.items():
    #     print(k, round(v[-1] / 100, 4))

    # plot_coverage(branch_cov_dict, "branches")
    # plot_coverage(line_cov_dict, "lines")
    # plot_coverage(func_cov_dict, "functions")

    plot_coverage_smooth(branch_cov_dict, "Branch")
    plot_coverage_smooth(line_cov_dict, "Line")
    # plot_coverage_smooth(func_cov_dict, "functions")
