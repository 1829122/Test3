import os
import json
import math
import torch
import random
import numpy as np
from scipy.sparse import csr_matrix

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')

def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item




def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items)) # (row, col) 위치에 data값 삽입

    return rating_matrix


def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item



def get_user_seqs(args):
    a_time_seq = []
    n_time_seq = []
    s_time_seq = []
    user_idxs = []
    print(args.n_time_file)
    time_lines = open(args.n_time_file).readlines()
    for line in time_lines:
        user, times = line.strip().split(' ', 1)
        user_idxs.append(int(user))
        times = times.split(' ')
        times = [int(time) for time in times]
        n_time_seq.append(times)
        a_time_seq.append(times)
    
    time_lines = open(args.s_time_file).readlines()
    for line in time_lines:
        user, times = line.strip().split(' ', 1)
        user_idxs.append(int(user))
        times = times.split(' ')
        times = [int(time) for time in times]
        s_time_seq.append(times)
        a_time_seq.append(times)

    item_lines1 = open(args.n_data_file).readlines()
    a_user_seq = []
    n_user_seq = []
    s_user_seq = []
    item_set = set()
    for line in item_lines1:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        n_user_seq.append(items)
        a_user_seq.append(items)
        item_set = item_set | set(items)
    item_lines2 = open(args.s_data_file).readlines()
    for line in item_lines2:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        a_user_seq.append(items)
        s_user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    item_lines = item_lines1 + item_lines2
    num_users = len(item_lines)
    not_aug_num = int(args.var_rank_not_aug_ratio * num_users)
    not_aug_users = user_idxs[:not_aug_num]
    num_items = max_item + 2
    n_valid_rating_matrix = generate_rating_matrix_valid(n_user_seq, num_users, num_items)
    n_test_rating_matrix = generate_rating_matrix_test(n_user_seq, num_users, num_items)
    s_valid_rating_matrix = generate_rating_matrix_valid(s_user_seq, num_users, num_items)
    s_test_rating_matrix = generate_rating_matrix_test(s_user_seq, num_users, num_items)
    valid_rating_matrix = generate_rating_matrix_valid(a_user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(a_user_seq, num_users, num_items)
    return n_user_seq, n_time_seq,s_user_seq, s_time_seq, max_item, valid_rating_matrix,test_rating_matrix , not_aug_users


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return [res / float(len(actual)), res, float(len(actual))]


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
    
def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return [sum_recall / true_users, sum_recall, true_users]


