import os
import json
import math
import torch
import random
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict

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
    return n_user_seq, n_time_seq,s_user_seq, s_time_seq, max_item, valid_rating_matrix,test_rating_matrix , not_aug_users, n_valid_rating_matrix,n_test_rating_matrix,s_valid_rating_matrix,s_test_rating_matrix  
def get_var(n_user_diff, s_user_diff):
    n_user_var = defaultdict(float)
    s_user_var = defaultdict(float)
    
    # n_user_diff의 각 인덱스별 합과 개수를 계산
    n_count = defaultdict(int)
    for idx, seq in enumerate(n_user_diff):
        for value in range(len(seq)):
            n_user_var[idx] += value
            n_count[idx] += 1
    
    # 인덱스별 평균 계산
    for idx in n_user_var:
        n_user_var[idx] /= n_count[idx]
    
    # s_user_diff의 경우 동일하게 처리
    s_count = defaultdict(int)
    for idx, seq in enumerate(s_user_diff):
        for value in range(len(seq)):
            s_user_var[idx] += value
            s_count[idx] += 1
            
    for idx in s_user_var:
        s_user_var[idx] /= s_count[idx]
    
    return n_user_var, s_user_var



def get_user_augmentation(n_user_seq, n_time_seq, s_user_seq, s_time_seq):
    n_user_diff = []
    s_user_diff = []
    n_user_aug = []
    s_user_aug = []
    n_time_aug = []
    s_time_aug = []

    for n_seq in n_time_seq:
        diff = []
        for idx in range(len(n_seq)):
            if idx == len(n_seq) - 1:
                break
            diff.append(n_seq[idx + 1] - n_seq[idx])

        n_user_diff.append(diff)
        diff = []
    
    for s_seq in s_time_seq:
        diff = []
        for idx in range(len(s_seq)):
            if idx == len(s_seq) - 1:
                break
            diff.append(s_seq[idx + 1] - s_seq[idx])
        s_user_diff.append(diff)
        diff = []

    n_user_var, s_user_var = get_var(n_user_diff, s_user_diff)

    for user_seq, time_seq, n_time in zip(n_user_seq, n_user_diff, n_time_seq):
        user_seq = user_seq[:-2]
        time_seq = time_seq[:-2]
        n_time = n_time[:-2]
        n_user_aug.append(user_seq)
        n_time_aug.append(n_time)
        a_user = []
        a_time = []
        user_new_list = []
        time_new_list = []
        user_new_list.append(user_seq[0])
        time_new_list.append(n_time[0])
        for idx in range(1, len(time_seq) + 1):
            if n_user_var[idx] > time_seq[idx - 1]:
                user_new_list.append(user_seq[idx])
                time_new_list.append(n_time[idx])
            else:
                user_new_list.append(user_seq[idx])
                time_new_list.append(n_time[idx])
                n_user_aug.append(user_new_list)
                n_time_aug.append(time_new_list)
                a_user.append(user_new_list)
                a_time.append(time_new_list)
                user_new_list = []
                time_new_list = []
                user_new_list.append(user_seq[idx])
                time_new_list.append(n_time[idx])
            if len(user_seq) == idx:
                if len(user_new_list) <= 1:
                    continue
                else:
                    n_user_aug.append(user_new_list)
                    n_time_aug.append(time_new_list)
                    a_user.append(user_new_list)
                    a_time.append(time_new_list)
        list1 = []
        list2 = []
        for idx2 in range(len(a_user) - 1):
            if idx2 == len(a_user) - 1:
                break
            n_user_aug.append(a_user[idx2] + a_user[idx2+1])
            n_time_aug.append(a_time[idx2] + a_time[idx2+1])
            #s_user_aug.append(a_user[idx2] + a_user[idx2+1])
            #s_time_aug.append(a_time[idx2] + a_time[idx2+1])
            
        
    for user_seq, time_seq, s_time in zip(s_user_seq, s_user_diff, s_time_seq):
        user_seq = user_seq[:-2]
        time_seq = time_seq[:-2]
        s_time = s_time[:-2]
        a_user = []
        a_time = []
        s_user_aug.append(user_seq)
        s_time_aug.append(s_time)
        user_new_list = []
        time_new_list = []
        user_new_list.append(user_seq[0])
        time_new_list.append(s_time[0])
        for idx in range(1, len(time_seq) + 1):
            if s_user_var[idx] > time_seq[idx - 1]:
                user_new_list.append(user_seq[idx])
                time_new_list.append(s_time[idx])
            else:
                user_new_list.append(user_seq[idx])
                time_new_list.append(s_time_seq[idx])
                s_user_aug.append(user_new_list)
                s_time_aug.append(time_new_list)
                a_user.append(user_new_list)
                a_time.append(time_new_list)
                user_new_list = []
                time_new_list = []
                user_new_list.append(user_seq[idx])
                time_new_list.append(s_time_seq[idx])
            if len(user_seq) == idx:
                if len(user_new_list) <= 1:
                    continue
                else:
                    s_user_aug.append(user_new_list)
                    s_time_aug.append(time_new_list)
                    a_user.append(user_new_list)
                    a_time.append(time_new_list)
        list1 = []
        list2 = []
        for idx2 in range(len(a_user) - 1):
            if idx2 == len(a_user) - 1:
                break
            s_user_aug.append(a_user[idx2] + a_user[idx2+1])
            s_time_aug.append(a_time[idx2] + a_time[idx2+1])
            
    return n_user_aug, n_time_aug, s_user_aug, s_time_aug

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


