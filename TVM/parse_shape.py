from z3 import Int, Solver, sat
import math


def adjust_static_ndims(ori_shape, target_dim):
    target_shape = []
    ori_dim = len(ori_shape)
    diff_num = target_dim - ori_dim
    if diff_num > 0:  # increase the dim
        target_shape = ori_shape
        for i in range(diff_num):
            target_shape.append(1)
        return target_shape
    elif diff_num < 0:  # decrease the dim
        diff_num = abs(diff_num)
        combine_dim_value = 1
        for i in range(diff_num+1):
            combine_dim_value *= ori_shape[i]
        if combine_dim_value > 0:
            target_shape.append(combine_dim_value)
        else:
            target_shape.append(-1)
        target_shape.extend(ori_shape[diff_num+1:])
        return target_shape
    else:  # the same
        return ori_shape


# def adjust_static_ndims(ori_shape, donor_dyna_shape):
#     from transform import parse_constraints
#     constraints, new_shape = parse_constraints.solve_dynamic_shape(ori_shape, donor_dyna_shape)
#     return new_shape


def adjust_dynamic_ndims(ori_shape, target_dim):
    ori_dim = len(ori_shape)
    diff_num = target_dim - ori_dim
    if diff_num > 0:  # increase the dim
        target_shape = ori_shape
        for i in range(diff_num):
            target_shape.append(1)
        return target_shape
    elif diff_num < 0:  # decrease the dim
        target_shape = ori_shape[:target_dim]
        return target_shape
    else:  # the same
        return ori_shape


if __name__ == '__main__':
    '''
    res = adjust_static_ndims([1,2,3], 5)
    print(res)
    res = adjust_static_ndims([1,2,3, 4], 2)
    print(res)

    res = adjust_dynamic_ndims([1, 'm', 3], 5)
    print(res)
    res = adjust_dynamic_ndims([-1, 2, 'm', 'n'], 2)
    print(res)
'''
    adjust_static_ndims([1,2,3, 4], ['m', 'n'])