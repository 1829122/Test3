import copy
import torch
import random

from torch.utils.data import Dataset


class DataSets(Dataset):
    def __init__(self, args, user_seq, time_seq, sim_model_type, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.time_seq = time_seq
        self.data_type = data_type
        if sim_model_type =='online':
            self.sim_model = args.online_sim_model
        else :
            self.sim_model = args.online_sim_model

        self.cl_aug = {''}



