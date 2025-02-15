import copy
import torch
import random
from torch.utils.data import Dataset
from data_augmentation import Crop, Mask, Reorder, Substitute, Insert, Random, CombinatorialEnumerate
from utils import neg_sample
class DataSets(Dataset):
    def __init__(self, args, user_seq, time_seq, not_aug_users=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.time_seq = time_seq
        self.data_type = data_type
        self.similarity_model = args.offline_similarity_model
        self.max_len = args.max_seq_length
        self.augmentations = {'crop': Crop(args.crop_mode, args.crop_rate),
                              'mask': Mask(args.mask_mode, args.mask_rate),
                              'reorder': Reorder(args.reorder_mode, args.reorder_rate),
                              'substitute': Substitute(self.similarity_model, args.substitute_mode,
                                                       args.substitute_rate),
                              'insert': Insert(self.similarity_model, args.insert_rate, args.max_insert_num_per_pos),
                              'random': Random(args, self.similarity_model),
                              'combinatorial_enumerate': CombinatorialEnumerate(args, self.similarity_model)}
        self.base_transform = self.augmentations[self.args.base_augment_type]
        self.total_train_users = 0
        self.model_warm_up_train_users = args.model_warm_up_epochs * len(user_seq)
        self.not_aug_users = [99999]
      
    def data_augmentation(self, user_seq, time_seq, not_aug=False):
        augmented_seq = []
        for i in range(2):
            aug_user_seq = self.base_transform(user_seq, time_seq)
            pad_len = self.max_len - len(aug_user_seq)
            aug_user_seq = [0] * pad_len + aug_user_seq
            aug_user_seq = aug_user_seq[-self.max_len:]
            cur_tensors = (torch.tensor(aug_user_seq, dtype=torch.long))
            augmented_seq.append(cur_tensors)
        return augmented_seq
    
    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in copied_input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(copied_input_ids)
        copied_input_ids = [0] * pad_len + copied_input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        copied_input_ids = copied_input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(copied_input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(copied_input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

        return cur_rec_tensors
    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]
        times = self.time_seq[index]
        input_times = times[:-2]

        self.total_train_users += 1

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use
        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]
        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        if self.data_type == "train":
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            cf_tensors_list = []
            not_aug = False
            # if n_views == 2, then it's downgraded to pair-wise contrastive learning
            total_augmentation_pairs = 1
            if self.total_train_users <= self.model_warm_up_train_users:
                total_augmentation_pairs = 0
            if (user_id in self.not_aug_users) and (self.total_train_users > self.model_warm_up_train_users):
                not_aug = True
            for i in range(total_augmentation_pairs):
                cf_tensors_list.append(self.data_augmentation(input_ids, input_times, not_aug))
            return cur_rec_tensors, cf_tensors_list
        elif self.data_type == 'valid':
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            return cur_rec_tensors
        else:
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            return cur_rec_tensors

    def __len__(self):
        """
        consider n_view of a single sequence as one sample
        """
        return len(self.user_seq)


