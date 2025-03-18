# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils import recall_at_k, ndcg_k, get_metric
from models import KMeans




class Trainer:
    def __init__(self, model, n_train_dataloader,
                 n_eval_dataloader,
                 n_test_dataloader,
                 s_train_dataloader,
                 s_eval_dataloader,
                 s_test_dataloader,
                 n_cluster_dataloader,
                 s_cluster_dataloader,
                 args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.model = model

        self.batch_size = self.args.batch_size
        self.sim=self.args.sim
        self.max_len = args.max_seq_length
        self.all_item_size = args.item_size
        self.equal_probs = torch.full((self.batch_size * self.max_len, self.all_item_size), 1.0 / self.all_item_size)

        cluster = KMeans(
            num_cluster=args.intent_num,
            seed=1,
            hidden_size=args.hidden_size,
            gpu_id=args.gpu_id,
            device=torch.device("cuda"),
        )
        self.clusters = [cluster]
        self.clusters_t=[self.clusters]

        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.n_cluster_dataloader = n_cluster_dataloader
        self.s_cluster_dataloader = s_cluster_dataloader
        self.n_train_dataloader = n_train_dataloader
        self.n_eval_dataloader = n_eval_dataloader
        self.n_test_dataloader = n_test_dataloader
        self.s_train_dataloader = s_train_dataloader
        self.s_eval_dataloader = s_eval_dataloader
        self.s_test_dataloader = s_test_dataloader
        self.n_train_matrix = args.n_train_matrix
        self.n_train_matrix = args.n_train_matrix 
        self.s_train_matrix = args.s_train_matrix 
        self.n_test_matrix = args.n_test_matrix 
        self.s_test_matrix = args.s_test_matrix 


        self.optim = Adam(self.model.parameters(), lr=self.args.lr,weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.n_train_dataloader, self.s_train_dataloader, self.n_train_matrix, self.s_train_matrix, self.n_cluster_dataloader, self.s_cluster_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.n_eval_dataloader, self.s_eval_dataloader, self.n_train_matrix, self.s_train_matrix, self.n_cluster_dataloader, self.s_cluster_dataloader, full_sort=full_sort, train=False, mode = 'valid')
    
    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.n_test_dataloader, self.s_test_dataloader, self.n_test_matrix, self.s_test_matrix, self.n_cluster_dataloader, self.s_cluster_dataloader, full_sort=full_sort, train=False, mode = 'test')

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": "{:.4f}".format(HIT_1),
            "NDCG@1": "{:.4f}".format(NDCG_1),
            "HIT@5": "{:.4f}".format(HIT_5),
            "NDCG@5": "{:.4f}".format(NDCG_5),
            "HIT@10": "{:.4f}".format(HIT_10),
            "NDCG@10": "{:.4f}".format(NDCG_10),
            "MRR": "{:.4f}".format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
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
        #if mode == 'test':
            #with open(self.args.test_log_file, 'a') as f:
                #f.write(str(post_fix) + '\n')
        return recall, ndcg, str(post_fix), save_bool
    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    # False Negative Mask
    def mask_correlated_samples_(self, label):
        label=label.view(1,-1)
        label=label.expand((2,label.shape[-1])).reshape(1,-1)
        label = label.contiguous().view(-1, 1)
        mask = torch.eq(label, label.t())
        return mask==0

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot',intent_id=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        if sim == 'cos':
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.t()) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        if self.args.f_neg:
            mask = self.mask_correlated_samples_(intent_id)
            negative_samples = sim
            negative_samples[mask==0]=float("-inf")
        else:
            mask = self.mask_correlated_samples(batch_size)
            negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def cicl_loss(self,coarse_intents,target_item):
        coarse_intent_1,coarse_intent_2=coarse_intents[0],coarse_intents[1]
        sem_nce_logits, sem_nce_labels = self.info_nce(coarse_intent_1[:, -1, :], coarse_intent_2[:, -1, :],
                                                       self.args.temperature, coarse_intent_1.shape[0], self.sim,
                                                       target_item[:, -1])
        cicl_loss = nn.CrossEntropyLoss()(sem_nce_logits, sem_nce_labels)
        return cicl_loss


    def ficl_loss(self, sequences, clusters_t):
        output = sequences[0][:,-1,:]
        intent_n = output.view(-1, output.shape[-1])  # [BxH]
        intent_n = intent_n.detach().cpu().numpy()
        intent_id, seq_to_v = clusters_t[0].query(intent_n)

        seq_to_v = seq_to_v.view(seq_to_v.shape[0], -1) # [BxH]
        a, b = self.info_nce(output.view(output.shape[0], -1), seq_to_v, self.args.temperature, output.shape[0], sim=self.sim,intent_id=intent_id)
        loss_n_0 = nn.CrossEntropyLoss()(a, b)

        output_s = sequences[1][:,-1,:]
        intent_n = output_s.view(-1, output_s.shape[-1])
        intent_n = intent_n.detach().cpu().numpy()
        intent_id, seq_to_v_1 = clusters_t[0].query(intent_n)  # [BxH]
        seq_to_v_1 = seq_to_v_1.view(seq_to_v_1.shape[0], -1) # [BxH]
        a, b = self.info_nce(output_s.view(output_s.shape[0], -1), seq_to_v_1, self.args.temperature, output_s.shape[0], sim=self.sim,intent_id=intent_id)
        loss_n_1 = nn.CrossEntropyLoss()(a, b)
        ficl_loss = loss_n_0 + loss_n_1

        return ficl_loss



class ICSRecTrainer(Trainer):
    def __init__(self, model, n_train_dataloader,
                 n_eval_dataloader,
                 n_test_dataloader,
                 s_train_dataloader,
                 s_eval_dataloader,
                 s_test_dataloader,
                 n_cluster_dataloader,
                 s_cluster_dataloader,
                 args):
        super(ICSRecTrainer, self).__init__(
                model, n_train_dataloader,
                n_eval_dataloader,
                n_test_dataloader,
                s_train_dataloader,
                s_eval_dataloader,
                s_test_dataloader,
                n_cluster_dataloader,
                s_cluster_dataloader,
                args)
    def mutual_kl_divergence(self, log_probs1, log_probs2):
        """
        log_probs1, log_probs2: 이미 log_softmax가 적용된 [B*seq_len, item_size] 형태
        """
        
        '''
        C = SASRec을 통과한 후 나온 벡터
        아이템이 총 2개라고 가정 e1, e2
        z1 = Ce1
        z2 = Ce2
        P = [p1, p2]
        p1 = e^z1 / e^z1 + e^z2
        p2 = e^z2 / e^z1 + e^z2
        Loss = PlogP -> -(-PlogP) -> -PlogP는 Shannon Entropy
        -> 클래스가(총 아이템) 2개라고 가정 -> 0.5,0.5일 때 Entropy가 max 
        -> Loss = -PlogP라고 생각 최소화한다고 생각한다면 확률이 극단적으로 가야지 Loss 감소 -> 아이템 임베딩 벡터가 극단적으로 학습
        -> Loss = PlogP라고 생각 -(-PlogP)이기 때문에 확률이 모일 수록 Loss 감소
        #####미분
        dp1/dz1 = p1(1 - p1) 
        dp2/dz1 = -p1p2
        dp2/dz2 = p2(1 - p2)
        dp1/dz2 = - p1p2
        dL / dp_k = 1 + logp_k
        dL / dz1 = dp1/dz1 * dL/dz1 + dp2/dz1 * dL/dp2
        dL / dz2 = 
        
        '''
        # 이미 log-softmax된 값이라면, 내부에서 또 log_softmax를 하면 안됨
        #p1 = log_probs1 Beauty_23
        p1 = log_probs1.exp()
        p2 = log_probs2
        kl_12 = (p2 * (torch.log(p2) - torch.log(p1))).sum(dim= -1)
        kl_21 = (p1 * (torch.log(p1) - torch.log(p2))).sum(dim= -1)
        #p1 = log_probs1.exp()  # shape = [B*seq_len, item_size]
        #p2 = log_probs2.exp()
        '''
        마이너스를 빼낸다면  Loss = -(PQ - PlogP)의 형태
        뒤에 PlogP는 정보 엔트로피의 식과 동일 해당 식을을
        '''
        #pq = (p2 * (torch.log(p2) - p1)).sum(dim = -1) # KL divergence XXXXX
        #pq = (p2 * (torch.log(p2) - torch.log(p1))).sum(dim = -1) # KL divergence입니다.
        # reduction='none'로 각 샘플별 kl 값을 얻고, 필요한 경우 마스킹으로 처리
        #kl_12 = kl_div(p1, p2, reduction='none').sum(dim=-1)  # [B*seq_len]
        #kl_21 = kl_div(p2, p1, reduction='none').sum(dim=-1)
        return kl_12 + kl_21  # [B*seq_len] 원소별 kl 값 합

    def iteration(self, epoch, n_dataloader, s_dataloader,n_train_matrix, s_train_matrix, n_cluster_dataloader, s_cluster_dataloader, full_sort=True, train=True, mode = 'train'):
        str_code = "train" if train else "test"
        if train:
            if self.args.cl_mode in ['cf','f']:
                # ------ intentions clustering ----- #
                print("Preparing Clustering:")
                self.model.eval()
                # save N
                kmeans_training_data = []
                rec_cf_data_iter = tqdm(enumerate(n_cluster_dataloader), total=len(n_cluster_dataloader))
                for i, (rec_batch) in rec_cf_data_iter:
                    """
                    rec_batch shape: key_name x batch_size x feature_dim
                    cl_batches shape: 
                        list of n_views x batch_size x feature_dim tensors
                    """
                    # 0. batch_data will be sent into the device(GPU or CPU)
                    rec_batch = tuple(t.to(self.device) for t in rec_batch)
                    _, subsequence, _, _, _ = rec_batch
                    n_sequence_output, s_sequence_output, sequence_output_a = self.model(subsequence) # [BxLxH]
                    sequence_output_b= n_sequence_output[:,-1,:] # [BxH]
                    kmeans_training_data.append(sequence_output_b.detach().cpu().numpy())

                kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)
                kmeans_training_data_t = [kmeans_training_data]

                for i, clusters in tqdm(enumerate(self.clusters_t), total=len(self.clusters_t)):
                    for j, cluster in enumerate(clusters):
                        cluster.train(kmeans_training_data_t[i])
                        self.clusters_t[i][j] = cluster

                # clean memory
                del kmeans_training_data
                del kmeans_training_data_t
                import gc
                gc.collect()

            # ------ model training -----#
            print("Performing Rec model Training:")
            self.model.train()
            rec_avg_loss = 0.0
            joint_avg_loss = 0.0
            icl_losses=0.0
            kl_avg_loss = 0.0
            print(f"rec dataset length: {len(n_dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(n_dataloader), total=len(n_dataloader))

            for i, (rec_batch) in rec_cf_data_iter:
                """             
                rec_batch shape: key_name x batch_size x feature_dim
                """
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, subsequence_1, target_pos_1, subsequence_2, _ = rec_batch

                # ---------- prediction task -----------------------#
                n_sequence_output, s_sequence_output,intent_output = self.model(subsequence_1)
                logits = self.predict_full(intent_output[:, -1, :])  #  [Bx|I|]
                rec_loss = nn.CrossEntropyLoss()(logits, target_pos_1[:, -1])

                # ---------- intent representation learning task ---------------#
                n_sequence_output, s_sequence_output, coarse_intent_1 = self.model(subsequence_1)
                n_sequence_output2, s_sequence_output2, coarse_intent_2 = self.model(subsequence_2)
                if self.args.cl_mode in ['c','cf']:
                    cicl_loss=self.cicl_loss([n_sequence_output,n_sequence_output2],target_pos_1)
                else:
                    cicl_loss=0.0
                if self.args.cl_mode in ['f','cf']:
                    ficl_loss = self.ficl_loss([n_sequence_output,n_sequence_output2], self.clusters_t[0])
                else:
                    ficl_loss = 0.0
                icl_loss = self.args.lambda_0 * cicl_loss + self.args.beta_0 * ficl_loss

                # ---------- multi-task learning --------------------#
                joint_loss =self.args.rec_weight*rec_loss+icl_loss
                
                B, seq_len, hidden_dim = n_sequence_output.size()
                # [num_items, hidden_dim]
                all_item_emb = self.model.item_embeddings.weight

                # [B*seq_len, hidden_dim]
                n_seq_flat = n_sequence_output.view(-1, hidden_dim)
                s_seq_flat = s_sequence_output.view(-1, hidden_dim)

                # [B*seq_len, num_items]
                n_logits = torch.matmul(n_seq_flat, all_item_emb.transpose(0,1))
                #s_logits = torch.matmul(s_seq_flat, all_item_emb.transpose(0,1))
                n_log_probs = F.log_softmax(n_logits, dim=-1)

                # 미리 생성된 equal_probs를 불러온 후 필요한 만큼 잘라줍니다.
                equal_probs = self.equal_probs[:B * seq_len].to(self.device)

                # mask 생성 및 적용
                mask = (subsequence_1 > 0).view(-1).float()
                masked_n_log_probs = n_log_probs[mask.bool()]
                masked_equal_probs = equal_probs[mask.bool()]

                # KL 계산 (reduction='none' 활용)
                kl_each_pos = self.mutual_kl_divergence(masked_n_log_probs, masked_equal_probs)

                # 평균 계산
                ml_loss = kl_each_pos.mean()
                joint_loss += ml_loss
                kl_avg_loss += ml_loss
                
                for param in self.model.item_encoder.parameters():
                    param.requires_grad = True
                for param in self.model.item_encoder2.parameters():
                    param.requires_grad = False
                self.model.position_embeddings.weight.requires_grad = True
                self.model.position_embedding2.weight.requires_grad = False

                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()
                if type(icl_loss)!=float:
                    icl_losses += icl_loss.item()
                else:
                    icl_losses+=icl_loss
                joint_avg_loss += joint_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_cf_data_iter)),
                "kl_avg_loss": "{:.4f}".format(kl_avg_loss / len(rec_cf_data_iter)),
                "icl_avg_loss": "{:.4f}".format(icl_losses/ len(rec_cf_data_iter)),
                "joint_avg_loss": "{:.4f}".format(joint_avg_loss / len(rec_cf_data_iter)),
            }
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")

            if self.args.cl_mode in ['cf','f']:
                # ------ intentions clustering ----- #
                print("Preparing Clustering:")
                self.model.eval()
                # save N
                kmeans_training_data = []
                rec_cf_data_iter = tqdm(enumerate(s_cluster_dataloader), total=len(s_cluster_dataloader))
                for i, (rec_batch) in rec_cf_data_iter:
                    """
                    rec_batch shape: key_name x batch_size x feature_dim
                    cl_batches shape: 
                        list of n_views x batch_size x feature_dim tensors
                    """
                    # 0. batch_data will be sent into the device(GPU or CPU)
                    rec_batch = tuple(t.to(self.device) for t in rec_batch)
                    _, subsequence, _, _, _ = rec_batch
                    n_sequence_output, s_sequence_output, sequence_output_a = self.model(subsequence) # [BxLxH]
                    sequence_output_b= s_sequence_output[:,-1,:] # [BxH]
                    kmeans_training_data.append(sequence_output_b.detach().cpu().numpy())

                kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)
                kmeans_training_data_t = [kmeans_training_data]

                for i, clusters in tqdm(enumerate(self.clusters_t), total=len(self.clusters_t)):
                    for j, cluster in enumerate(clusters):
                        cluster.train(kmeans_training_data_t[i])
                        self.clusters_t[i][j] = cluster

                # clean memory
                del kmeans_training_data
                del kmeans_training_data_t
                import gc
                gc.collect()

            # ------ model training -----#
            print("Performing Rec model Training:")
            self.model.train()
            rec_avg_loss = 0.0
            joint_avg_loss = 0.0
            icl_losses=0.0
            kl_avg_loss = 0.0
            print(f"rec dataset length: {len(s_dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(s_dataloader), total=len(s_dataloader))

            for i, (rec_batch) in rec_cf_data_iter:
                """             
                rec_batch shape: key_name x batch_size x feature_dim
                """
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, subsequence_1, target_pos_1, subsequence_2, _ = rec_batch

                # ---------- prediction task -----------------------#
                n_sequence_output, s_sequence_output, intent_output = self.model(subsequence_1)
                logits = self.predict_full(intent_output[:, -1, :])  #  [Bx|I|]
                rec_loss = nn.CrossEntropyLoss()(logits, target_pos_1[:, -1])

                # ---------- intent representation learning task ---------------#
                n_sequence_output, s_sequence_output, coarse_intent_1 = self.model(subsequence_1)
                n_sequence_output2, s_sequence_output2, coarse_intent_2 = self.model(subsequence_2)
                if self.args.cl_mode in ['c','cf']:
                    cicl_loss=self.cicl_loss([ s_sequence_output, s_sequence_output2],target_pos_1)
                else:
                    cicl_loss=0.0
                if self.args.cl_mode in ['f','cf']:
                    ficl_loss = self.ficl_loss([s_sequence_output, s_sequence_output2], self.clusters_t[0])
                else:
                    ficl_loss = 0.0
                icl_loss = self.args.lambda_0 * cicl_loss + self.args.beta_0 * ficl_loss

                # ---------- multi-task learning --------------------#
                joint_loss =self.args.rec_weight*rec_loss+icl_loss
                
                B, seq_len, hidden_dim = n_sequence_output.size()
                # [num_items, hidden_dim]
                all_item_emb = self.model.item_embeddings.weight

                # [B*seq_len, hidden_dim]
                #n_seq_flat = n_sequence_output.view(-1, hidden_dim)
                s_seq_flat = s_sequence_output.view(-1, hidden_dim)

                # [B*seq_len, num_items]
                s_logits = torch.matmul(s_seq_flat, all_item_emb.transpose(0,1))
                #s_logits = torch.matmul(s_seq_flat, all_item_emb.transpose(0,1))
                s_log_probs = F.log_softmax(s_logits, dim=-1)

                # 미리 생성된 equal_probs를 불러온 후 필요한 만큼 잘라줍니다.
                equal_probs = self.equal_probs[:B * seq_len].to(self.device)

                # mask 생성 및 적용
                mask = (subsequence_1 > 0).view(-1).float()
                masked_n_log_probs = s_log_probs[mask.bool()]
                masked_equal_probs = equal_probs[mask.bool()]

                # KL 계산 (reduction='none' 활용)
                kl_each_pos = self.mutual_kl_divergence(masked_n_log_probs, masked_equal_probs)

                # 평균 계산
                ml_loss = kl_each_pos.mean()
                joint_loss += ml_loss
                kl_avg_loss += ml_loss
                
                for param in self.model.item_encoder.parameters():
                    param.requires_grad = False
                for param in self.model.item_encoder2.parameters():
                    param.requires_grad = True
                self.model.position_embeddings.weight.requires_grad = False
                self.model.position_embedding2.weight.requires_grad = True
                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()
                if type(icl_loss)!=float:
                    icl_losses += icl_loss.item()
                else:
                    icl_losses+=icl_loss
                joint_avg_loss += joint_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_cf_data_iter)),
                "kl_avg_loss": "{:.4f}".format(kl_avg_loss / len(rec_cf_data_iter)),
                "icl_avg_loss": "{:.4f}".format(icl_losses/ len(rec_cf_data_iter)),
                "joint_avg_loss": "{:.4f}".format(joint_avg_loss / len(rec_cf_data_iter)),
            }
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")

        else:
            rec_data_iter = tqdm(enumerate(n_dataloader), total=len(n_dataloader))
            self.model.eval()
            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, answers = batch
                    n_sequence_output, s_sequence_output, sequence_output = self.model(input_ids)
                    sequence_output = sequence_output[:,-1,:]
                    #n_recommend_output = recommend_output
                    #n_recommend_output = n_recommend_output[:,-1,:]
                    # recommendation results

                    rating_pred = self.predict_full(sequence_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[n_train_matrix[batch_user_index].toarray() > 0] = 0
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

                rec_data_iter = tqdm(enumerate(s_dataloader), total=len(s_dataloader))
                self.model.eval()
                pred_list = None
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, answers = batch
                    n_sequence_output, s_sequence_output, sequence_output = self.model(input_ids)
                    sequence_output = sequence_output[:,-1,:]
                    #s_recommend_output = recommend_output
                    #s_recommend_output = s_recommend_output[:, -1, :]
                    # recommendation results

                    rating_pred = self.predict_full(sequence_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[s_train_matrix[batch_user_index].toarray() > 0] = 0
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
                with open(self.args.log_file, 'a') as f:
                    f.write(str(post_fix) + '\n')
                
                
                return final_recall, final_ndcg

            else:
                for i, batch in rec_data_iter:
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
