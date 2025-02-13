import copy
import random
import itertools
import numpy as np



def get_var(tlist):
    length = len(tlist)
    total = 0
    diffs = []

    if length == 1:
        return 0

    for i in range(length - 1):
        diff = abs(tlist[i + 1] - tlist[i])
        diffs.append(diff)
        total = total + diff
    avg_diff = total / len(diffs)

    total = 0
    for diff in diffs:
        total = total + (diff - avg_diff) ** 2
    result = total / len(diffs)

    return result



class Crop(object):
    def __init__(self, mode, ratio = 0.2):
        self.ratio = ratio
        self.mode = mode



    def __call__(self, item_seq, time_seq):
        copied_sequence = copy.deepcopy(item_seq)
        sub_seq_length = int(self.tao * len(copied_sequence))
        # randint generate int x in range: a <= x <= b
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        if sub_seq_length <= 2:
            return copied_sequence
        
        crop_var = []
        crop_index = []
        for i in range(len(item_seq)):
            if len(item_seq) - i - sub_seq_length >= 0:
                left_index = len(item_seq) - i - sub_seq_length
                right_index = left_index + sub_seq_length - 1
                time_index = time_seq[left_index:right_index]
                time_var = get_var(time_index)
                crop_var.append(time_var)
                crop_index.append(left_index)

        if self.mode == 'max':
            idx = crop_var.index(max(crop_var))
        else:
            idx = crop_var.index(min(crop_var))
        
        start_index = crop_index.index(idx)
        
        cropped_sequence = copied_sequence[start_index:start_index + sub_seq_length]
        return cropped_sequence


class Mask(object):
    def __init__(self, mode, ratio = 0.7):
        self.mode = mode
        self.ratio = ratio

    def __call__(self, item_seq, time_seq):
        copied_sequence = copy.deepcopy(item_seq)
        mask_num = int(len(item_seq) * self.ratio)
        mask = [0 for i in range(mask_num)]

        if len(copied_sequence) <= 1:
            return copied_sequence
        
        time_var = []
        length = len(time_seq)
        for i in range(length - 1):
            diff = abs(time_seq[i+1] - time_seq[i])
            time_var.append(diff)

        diff_sorted = []

        if self.mode == 'random':
            copied_sequence = copy.deepcopy(item_seq)
            mask_nums = int(self.gamma * len(copied_sequence))
            mask = [0 for i in range(mask_nums)]
            mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
            for idx, mask_value in zip(mask_idx, mask):
                copied_sequence[idx] = mask_value
            return copied_sequence
        if self.mode == 'maximum':
            """
            First sort from large to small, and then return the original index value by sorting. 
            The larger the value, the smaller the index value
            """
            diff_sorted = np.argsort(time_var)[::-1]
        if self.mode == 'minimum':
            """
            First sort from small to large, and then return the original index value by sorting. 
            The larger the value, the larger the index.
            """
            diff_sorted = np.argsort(time_var)
        diff_sorted = diff_sorted.tolist()
        mask_idx = []
        for i in range(mask_nums):
            temp = diff_sorted[i]
            mask_idx.append(temp)

        for idx, mask_value in zip(mask_idx, mask):
            copied_sequence[idx] = mask_value
        return copied_sequence
    
class Insert_Random(object):
    def __init__(self, mode, ratio, args):
        self.mode = mode
        self.ratio = ratio
        self.maxlen = args.maxlen

    def __call__(self, item_seq, time_seq):
        copied_seq = copy.deepcopy(item_seq)
        insert_nums = int(len(copied_seq) * self.ratio)
        item = None
        time_var = []

        if len(copied_seq) == self.maxlen:
            return copied_seq

        for i in range(len(time_seq) - 1):
            temp = abs(time_seq[i+1] - time_seq[i])
            time_var.append(temp)

        diff_sorted = np.argsort(time_var)[::-1]

        for i in range(insert_nums):
            copied_seq.insert(diff_sorted[i] + 2, item)
            if len(copied_seq) == self.maxlen:
                return copied_seq

        return copied_seq
    
class Insert(object): # 유사도 함수 추가
    def __init__(self, mode, ratio, args):
        self.mode = mode
        self.ratio = ratio
        self.maxlen = args.maxlen

    def __call__(self, item_seq, time_seq):
        copied_seq = copy.deepcopy(item_seq)
        insert_nums = int(len(copied_seq) * self.ratio)
        item = None
        time_var = []

        if len(copied_seq) == self.maxlen:
            return copied_seq

        for i in range(len(time_seq) - 1):
            temp = abs(time_seq[i+1] - time_seq[i])
            time_var.append(temp)

        if self.mode == 'max':
            diff_sorted = np.argsort(time_var)[::-1]

        else:
            diff_sorted = np.argsort(time_var)

        for i in range(insert_nums):
            for i in range(insert_nums):
                copied_seq.insert(diff_sorted[i] + 2, item)
            if len(copied_seq) == self.maxlen:
                return copied_seq

        return copied_seq


    


        
















