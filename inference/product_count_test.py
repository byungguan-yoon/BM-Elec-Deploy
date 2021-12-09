def v_sum(h_pro_tf, pro_is):
    result = [None for _ in range(16)]
    for i, (h_pro_tf_e, pro_is_e) in enumerate(zip(h_pro_tf, pro_is)):
        # h_pro_tf_e ex) [False, False, False]
        if i == 0:
            # h_pro_tf_e_e ex) False
            for j, h_pro_tf_e_e in enumerate(h_pro_tf_e):
                result[j] = h_pro_tf_e_e
        else:
            idx = find_none_idx(result) - (pro_is_e + 1)
            result = insert_result(result, h_pro_tf_e, idx)
    return result

def find_none_idx(result):
    for i, result_e in enumerate(result):
        if result_e == None:
            idx = i
            break
    return idx

def insert_result(result, h_pro_tf_e, idx):
    for i, h_pro_tf_e_e in enumerate(h_pro_tf_e):
        if result[idx + i] == None:
            result[idx + i] = h_pro_tf_e_e
        else:
            result[idx + i] = result[idx + i] or h_pro_tf_e_e
    return result


if __name__ == '__main__':
    h_pro_tf = [[False, False, False],
    [False, False, False, False],
    [False, False, False, False],
    [True, False, False, False],
    [False, False, False, False],
    [False, False, False, False],
    [False, False, False, False],
    [False, False, False]]
    pro_is = [1, 1, 1, 1, 1, 1, 1, 1]
    print(v_sum(h_pro_tf, pro_is))