# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam

from modules import NCELoss, NTXent
from torch.utils.data import DataLoader, RandomSampler
from datasets import DataSets
from utils import recall_at_k, ndcg_k, get_metric, get_user_seqs, nCr


class Trainer:
    def __init__(self, model, n_train_dataloader,
                 n_eval_dataloader,
                 n_test_dataloader,
                 s_train_dataloader,
                 s_eval_dataloader,
                 s_test_dataloader,
                 args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.HR = 0
        self.NDCG = 0
        self.model = model
        self.online_similarity_model = None

        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)
        # projection head for contrastive learn task
        self.projection = nn.Sequential(nn.Linear(self.args.max_seq_length * self.args.hidden_size, \
                                                  512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                        nn.Linear(512, self.args.hidden_size, bias=True))
        if self.cuda_condition:
            self.model.cuda()
            self.projection.cuda()
        # Setting the train and test data loader
        self.n_train_dataloader = n_train_dataloader
        self.n_eval_dataloader = n_eval_dataloader
        self.n_test_dataloader = n_test_dataloader
        self.s_train_dataloader = s_train_dataloader
        self.s_eval_dataloader = s_eval_dataloader
        self.s_test_dataloader = s_test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        # self.cf_criterion = NTXent()
        print("self.cf_criterion:", self.cf_criterion.__class__.__name__)

    def __refresh_training_dataset(self, item_embedding):
        """
        use for updating item embedding
        """
        user_seq, time_seq, _, _, _, not_aug_users = get_user_seqs(self.args)
        self.args.online_similarity_model.update_embedding_matrix(item_embedding)
        # training data for node classification
        train_dataset = DataSets(self.args, user_seq, time_seq, not_aug_users=not_aug_users,
                                                          data_type='train', similarity_model_type='hybrid',
                                                          total_train_users=self.args.model_warm_up_epochs * len(user_seq)+1)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
        return train_dataloader

    def train(self, epoch):
        
        self.iteration(epoch, self.n_train_dataloader, self.s_train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.n_eval_dataloader, self.s_eval_dataloader,full_sort=full_sort, mode='valid')
    
    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.n_test_dataloader, self.s_test_dataloader,full_sort=full_sort, mode='test')
   
    def iteration(self, epoch, dataloader, full_sort=False, mode='train'):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list, mode):
        save_bool = False
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "stage": mode,
            "epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0][0]), "NDCG@5": '{:.4f}'.format(ndcg[0][0]),
            "HIT@10": '{:.4f}'.format(recall[1][0]), "NDCG@10": '{:.4f}'.format(ndcg[1][0]),
            "HIT@20": '{:.4f}'.format(recall[3][0]), "NDCG@20": '{:.4f}'.format(ndcg[3][0])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        if mode == 'test':
            with open(self.args.test_log_file, 'a') as f:
                f.write(str(post_fix) + '\n')
        return recall, ndcg, str(post_fix), save_bool
    

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embedding(pos_ids)
        neg_emb = self.model.item_embedding(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embedding(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embedding.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class CoSeRecTrainer(Trainer):

    def __init__(self, model, n_train_dataloader,
                 n_eval_dataloader,
                 n_test_dataloader,
                 s_train_dataloader,
                 s_eval_dataloader,
                 s_test_dataloader,
                 args):
        super(CoSeRecTrainer, self).__init__(
                model, n_train_dataloader,
                n_eval_dataloader,
                n_test_dataloader,
                s_train_dataloader,
                s_eval_dataloader,
                s_test_dataloader,
                args)
        

    def _one_pair_contrastive_learning(self, inputs):
        """
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        """
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        n_cl_sequence_output, s_cl_sequence_output = self.model.transformer_encoder(cl_batch)
        # cf_sequence_output = cf_sequence_output[:, -1, :]
        n_cl_sequence_flatten = n_cl_sequence_output.view(cl_batch.shape[0], -1)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0] // 2
        n_cl_output_slice = torch.split(n_cl_sequence_flatten, batch_size)
        n_cl_loss = self.cf_criterion(n_cl_output_slice[0],
                                    n_cl_output_slice[1])
        s_cl_sequence_flatten = s_cl_sequence_output.view(cl_batch.shape[0], -1)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0] // 2
        s_cl_output_slice = torch.split(s_cl_sequence_flatten, batch_size)
        s_cl_loss = self.cf_criterion(s_cl_output_slice[0],
                                    s_cl_output_slice[1])
        return n_cl_loss, s_cl_loss

    def iteration(self, epoch, n_dataloader, s_dataloader,full_sort=True, mode='train'):
        if mode == 'train':
            self.model.train()
            rec_avg_loss = 0.0
            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0

            print(f"rec dataset length: {len(n_dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(n_dataloader), total=len(n_dataloader))

            for i, (rec_batch, cl_batches) in rec_cf_data_iter:
                '''
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape: 
                    list of n_views x batch_size x feature_dim tensors
                '''
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                # ---------- recommendation task ---------------#
                n_sequence_output, s_sequence_output = self.model.transformer_encoder(input_ids)
                n_rec_loss = self.cross_entropy(n_sequence_output, target_pos, target_neg)
                s_rec_loss = self.cross_entropy(s_sequence_output, target_pos, target_neg)

                # ---------- contrastive learning task -------------#
                n_cl_losses = []
                s_cl_losses = []
                for cl_batch in cl_batches:
                    n_cl_loss, s_cl_loss = self._one_pair_contrastive_learning(cl_batch)
                    n_cl_losses.append(n_cl_loss)

                joint_loss = self.args.rec_weight * n_rec_loss
                joint_loss += 0.1 * s_rec_loss
                for cl_loss in n_cl_losses:
                    joint_loss += self.args.cf_weight * cl_loss
                for param in self.model.trm_encoder2.parameters():
                    param.requires_grad = False
                for param in self.model.trm_encoder.parameters():
                    param.requires_grad = True
                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += n_rec_loss.item()

                for i, cl_loss in enumerate(n_cl_losses):
                    cl_individual_avg_losses[i] += cl_loss.item()
                    cl_sum_avg_loss += cl_loss.item()
                joint_avg_loss += joint_loss.item()

            post_fix = {
            "epoch": epoch,
            "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_cf_data_iter)),
            "joint_avg_loss": '{:.4f}'.format(joint_avg_loss / len(rec_cf_data_iter)),
            "cl_avg_loss": '{:.4f}'.format(
                cl_sum_avg_loss / (len(rec_cf_data_iter) * self.total_augmentaion_pairs)),
            }
            for i, cl_individual_avg_loss in enumerate(cl_individual_avg_losses):
                post_fix['cl_pair_' + str(i) + '_loss'] = '{:.4f}'.format(
                    cl_individual_avg_loss / len(rec_cf_data_iter))

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))
            rec_avg_loss = 0.0
            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0
            rec_cf_data_iter = tqdm(enumerate(s_dataloader), total=len(s_dataloader))
            for i, (rec_batch, cl_batches) in rec_cf_data_iter:
                '''
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape: 
                    list of n_views x batch_size x feature_dim tensors
                '''
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                # ---------- recommendation task ---------------#
                n_sequence_output, s_sequence_output = self.model.transformer_encoder(input_ids)
                n_rec_loss = self.cross_entropy(n_sequence_output, target_pos, target_neg)
                s_rec_loss = self.cross_entropy(s_sequence_output, target_pos, target_neg)

                # ---------- contrastive learning task -------------#
                cl_losses = []
                n_cl_losses = []
                s_cl_losses = []
                for cl_batch in cl_batches:
                    n_cl_loss, s_cl_loss = self._one_pair_contrastive_learning(cl_batch)
                    s_cl_losses.append(s_cl_loss)

                joint_loss = self.args.rec_weight * s_rec_loss
                joint_loss += 0.1 * n_rec_loss
                for cl_loss in s_cl_losses:
                    joint_loss += self.args.cf_weight * cl_loss
                for param in self.model.trm_encoder2.parameters():
                    param.requires_grad = True
                for param in self.model.trm_encoder.parameters():
                    param.requires_grad = False
                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += s_rec_loss.item()

                for i, cl_loss in enumerate(s_cl_losses):
                    cl_individual_avg_losses[i] += cl_loss.item()
                    cl_sum_avg_loss += cl_loss.item()
                joint_avg_loss += joint_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_cf_data_iter)),
                "joint_avg_loss": '{:.4f}'.format(joint_avg_loss / len(rec_cf_data_iter)),
                "cl_avg_loss": '{:.4f}'.format(
                    cl_sum_avg_loss / (len(rec_cf_data_iter) * self.total_augmentaion_pairs)),
            }
            for i, cl_individual_avg_loss in enumerate(cl_individual_avg_losses):
                post_fix['cl_pair_' + str(i) + '_loss'] = '{:.4f}'.format(
                    cl_individual_avg_loss / len(rec_cf_data_iter))

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            rec_data_iter = tqdm(enumerate(n_dataloader),
                                 desc="Recommendation EP_%s:%d" % (mode, epoch),
                                 total=len(n_dataloader),
                                 bar_format="{l_bar}{r_bar}")
            self.model.eval()
            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.transformer_encoder(input_ids)

                    n_recommend_output, s_recommend_output = recommend_output
                    n_recommend_output = n_recommend_output[:,-1,:]
                    # recommendation results

                    rating_pred = self.predict_full(n_recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                n_recall_result, n_ndcg_result,_, _ = self.get_full_sort_score(epoch, answer_list, pred_list, mode)

                rec_data_iter = tqdm(enumerate(s_dataloader),
                                    desc="Recommendation EP_%s:%d" % (mode, epoch),
                                    total=len(s_dataloader),
                                    bar_format="{l_bar}{r_bar}")
                self.model.eval()
                pred_list = None
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.transformer_encoder(input_ids)

                    n_recommend_output, s_recommend_output = recommend_output
                    s_recommend_output = s_recommend_output[:, -1, :]
                    # recommendation results

                    rating_pred = self.predict_full(s_recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                s_recall_result, s_ndcg_result,_, _ = self.get_full_sort_score(epoch, answer_list, pred_list, mode)
                final_recall = []
                final_ndcg = []

                for i in range(len(n_recall_result)):
                    total_sum_recall = n_recall_result[i][1] + s_recall_result[i][1]
                    total_true_users = n_recall_result[i][2] + s_recall_result[i][2]
                    final_recall.append(total_sum_recall / total_true_users)

                    # sum_ndcg을 합치고, true_users를 합쳐서 새로운 ndcg 계산
                    total_sum_ndcg = n_ndcg_result[i][1] + s_ndcg_result[i][1]
                    total_true_users_ndcg = n_ndcg_result[i][2] + s_ndcg_result[i][2]
                    final_ndcg.append(total_sum_ndcg / total_true_users_ndcg)

                post_fix = {
                    "stage": mode,
                    "epoch": epoch,
                    "HIT@5": '{:.4f}'.format(final_recall[0]), "NDCG@5": '{:.4f}'.format(final_ndcg[0]),
                    "HIT@10": '{:.4f}'.format(final_recall[1]), "NDCG@10": '{:.4f}'.format(final_ndcg[1]),
                    "HIT@20": '{:.4f}'.format(final_recall[3]), "NDCG@20": '{:.4f}'.format(final_ndcg[3])
                }
                print(post_fix)
                if mode =='test':
                    with open(self.args.test_log_file, 'a') as f:
                        f.write(str(post_fix) + '\n')
                # 최종 결과 출력
                return final_recall, final_ndcg
                


                

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)
