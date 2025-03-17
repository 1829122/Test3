# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import kl_div, log_softmax, cross_entropy, cosine_similarity, pairwise_distance, relu
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
        self.batch_size = args.batch_size
        self.max_len = args.max_seq_length
        self.all_item_size = args.item_size
        self.equal_probs = torch.full((self.batch_size * self.max_len, self.all_item_size), 1.0 / self.all_item_size)

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
        #if mode == 'test':
            #with open(self.args.test_log_file, 'a') as f:
                #f.write(str(post_fix) + '\n')
        return recall, ndcg, str(post_fix), save_bool
    

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
    
    def bpr_loss(self, seq_out, pos_ids, neg_ids):  # BPR Loss
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embedding(pos_ids)
        neg_emb = self.model.item_embedding(neg_ids)
        
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)  # [batch*seq_len]
        
        # BPR Loss는 pos_logits가 neg_logits보다 크게 만드는 것이 목표
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        
        # BPR Loss 계산
        loss = -torch.sum(
            torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)
        
        return loss
    
    '''
    def cross_entropy(self, seq_out, pos_ids, neg_ids): # BCE
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
    '''
    
    def cross_entropy(self, seq_out, pos_ids, neg_ids): #CE
        """
        seq_out : [batch, seq_len, hidden_size]
        pos_ids : [batch, seq_len], 0은 패딩(무시할 아이템), 그 외는 실제 정답 아이템 ID
        """
        # 아이템 임베딩 가중치 (보통 [num_items, hidden_size] 형태)
        # self.model.item_embedding.weight : [num_items, hidden_size]
        
        # (1) seq_out과 item_embedding.weight^T 내적 => [batch, seq_len, num_items]
        logits = torch.matmul(seq_out, self.model.item_embedding.weight.transpose(0, 1))
        
        # (2) cross_entropy 계산
        #  - logits: [batch, seq_len, num_items] => [batch*seq_len, num_items]
        #  - pos_ids: [batch, seq_len] => [batch*seq_len]
        #  - ignore_index=0 : pos_ids가 0(패딩)이면 무시
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)),      # (batch*seq_len, num_items)
            pos_ids.view(-1),                     # (batch*seq_len)
            ignore_index=0
        )
        return loss
    '''
    def cross_entropy(self, seq_output, answers, neg_answers): # 데이터 증강강
        
        seq_output = seq_output[:, -1, :]
        item_emb = self.model.item_embedding.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)

        return loss
    '''  
    def cosine_sim(self, n_seq, s_seq):
        n_seq = n_seq[:, -1, :]
        s_seq = s_seq[:, -1, :]
        sim = cosine_similarity(n_seq, s_seq, dim = -1)
    
        sim = sim.view(-1)
        sim = 1 - sim
        sim = sim.mean()
        
        return sim

    def euclidean_contrastive_loss(self, n_sequence_output, s_sequence_output, margin=1.0):
        """
        유클리드 거리를 사용한 Contrastive Loss
        
        Args:
            n_sequence_output (torch.Tensor): [B, L, D] 크기의 인코더 1 출력
            s_sequence_output (torch.Tensor): [B, L, D] 크기의 인코더 2 출력
            margin (float): Negative pair에 적용할 최소 거리
        
        Returns:
            torch.Tensor: Contrastive Loss 값
        """
        B, L, D = n_sequence_output.shape
        N = B * L
        
        n_seq_flat = n_sequence_output.view(N, D)  # [N, D]
        s_seq_flat = s_sequence_output.view(N, D)  # [N, D]

        # 같은 위치의 벡터 쌍의 유클리드 거리 계산 (Positive pairs)
        positive_distances = pairwise_distance(n_seq_flat, s_seq_flat, p=2)  # [N]

        # Negative pairs: 배치 내 모든 다른 쌍과의 거리 계산
        all_distances = torch.cdist(n_seq_flat, s_seq_flat, p=2)  # [N, N]

        # Contrastive Loss 계산
        positive_loss = (positive_distances ** 2).mean()
        negative_loss = relu(margin - all_distances).mean()

        return 0.01 * (positive_loss + negative_loss)
    
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
        n_cl_sequence_output, s_cl_sequence_output, t_cl_sequence_output = self.model.transformer_encoder(cl_batch)
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
        t_cl_sequence_flatten = t_cl_sequence_output.view(cl_batch.shape[0], -1)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0] // 2
        t_cl_output_slice = torch.split(t_cl_sequence_flatten, batch_size)
        t_cl_loss = self.cf_criterion(t_cl_output_slice[0],
                                    t_cl_output_slice[1])
        return n_cl_loss, s_cl_loss, t_cl_loss
    
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

    def iteration(self, epoch, n_dataloader, s_dataloader,full_sort=True, mode='train'):
        if mode == 'train':
            self.model.train()
            rec_avg_loss = 0.0
            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0
            kl_avg_loss = 0.0
            sim_avg_loss = 0.0
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
                #print(target_neg.shape)

                # ---------- recommendation task ---------------#
                n_sequence_output, s_sequence_output, t_sequence_output = self.model.transformer_encoder(input_ids)
                '''
                n_rec_loss = self.bpr_loss(n_sequence_output, target_pos, target_neg)
                s_rec_loss = self.bpr_loss(s_sequence_output, target_pos, target_neg)
                t_rec_loss = self.bpr_loss(t_sequence_output, target_pos, target_neg)
                '''
                n_rec_loss = self.cross_entropy(n_sequence_output, target_pos, target_neg)
                #s_rec_loss = self.cross_entropy(s_sequence_output, target_pos, target_neg)
                #t_rec_loss = self.cross_entropy(t_sequence_output, target_pos, target_neg)
                
                #print(n_sequence_output.shape)
                # ---------- contrastive learning task -------------#
                n_cl_losses = []
                s_cl_losses = []
                for cl_batch in cl_batches:
                    n_cl_loss, s_cl_loss, t_cl_loss = self._one_pair_contrastive_learning(cl_batch)
                    n_cl_losses.append(n_cl_loss)
                
                joint_loss = self.args.rec_weight * n_rec_loss
                #mask = (target_pos > 0).view(-1).float()
                #sim_loss = self.cosine_sim(n_sequence_output, s_sequence_output)
                #sim_loss = (sim_loss * mask).sum() / (mask.sum() + 1e-9)
                #joint_loss += sim_loss
                #sim_avg_loss += sim_loss
                
                B, seq_len, hidden_dim = n_sequence_output.size()
                # [num_items, hidden_dim]
                all_item_emb = self.model.item_embedding.weight

                # [B*seq_len, hidden_dim]
                n_seq_flat = n_sequence_output.view(-1, hidden_dim)
                s_seq_flat = s_sequence_output.view(-1, hidden_dim)

                # [B*seq_len, num_items]
                n_logits = torch.matmul(n_seq_flat, all_item_emb.transpose(0,1))
                #s_logits = torch.matmul(s_seq_flat, all_item_emb.transpose(0,1))
                n_log_probs = log_softmax(n_logits, dim=-1)

                # 미리 생성된 equal_probs를 불러온 후 필요한 만큼 잘라줍니다.
                equal_probs = self.equal_probs[:B * seq_len].to(self.device)

                # mask 생성 및 적용
                mask = (input_ids > 0).view(-1).float()
                masked_n_log_probs = n_log_probs[mask.bool()]
                masked_equal_probs = equal_probs[mask.bool()]

                # KL 계산 (reduction='none' 활용)
                kl_each_pos = self.mutual_kl_divergence(masked_n_log_probs, masked_equal_probs)

                # 평균 계산
                ml_loss = kl_each_pos.mean()
                joint_loss += ml_loss
                kl_avg_loss += ml_loss
                
                #sim_loss = self.cosine_sim(n_sequence_output, s_sequence_output)
                #joint_loss += sim_loss
                #sim_avg_loss += sim_loss
                
                
                
                '''
                # log softmax
                n_log_probs = log_softmax(n_logits, dim=-1)
                #n_log_probs = torch.sigmoid(n_logits)
                #n_log_probs = log_softmax(n_log_probs, dim=-1)
                #s_log_probs = log_softmax(s_logits, dim=-1)
                equal_probs = torch.full((B * seq_len, all_item_emb.size(0)), 1.0 / all_item_emb.size(0))
                equal_probs = equal_probs.to(self.device)
                # target_pos가 0인 곳은 (학습 시점이 없는) 패딩
                # shape = [B, seq_len] ->  [B*seq_len]
                mask = (target_pos > 0).view(-1).float()

                # KL 계산 (reduction='none' 활용)
                kl_each_pos = self.mutual_kl_divergence(n_log_probs, equal_probs)  # [B*seq_len]
                # 패딩 위치 제외 후 평균
                ml_loss = (kl_each_pos * mask).sum() / (mask.sum() + 1e-9)
                joint_loss += ml_loss
                kl_avg_loss += ml_loss
                '''
                
                for cl_loss in n_cl_losses:
                    joint_loss += self.args.cf_weight * cl_loss
                for param in self.model.trm_encoder2.parameters():
                    param.requires_grad = False
                for param in self.model.trm_encoder.parameters():
                    param.requires_grad = True
                self.model.position_embedding.weight.requires_grad = True
                self.model.position_embedding2.weight.requires_grad = False
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
            "kl_avg_loss": '{:.4f}'.format(kl_avg_loss / len(rec_cf_data_iter)),
            "sim_avg_loss": '{:.4f}'.format(sim_avg_loss / len(rec_cf_data_iter)),
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
            kl_avg_loss = 0.0
            sim_avg_loss = 0.0
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
                #print(target_neg.shape)

                # ---------- recommendation task ---------------#
                n_sequence_output, s_sequence_output, t_sequence_output = self.model.transformer_encoder(input_ids)
                '''
                n_rec_loss = self.bpr_loss(n_sequence_output, target_pos, target_neg)
                s_rec_loss = self.bpr_loss(s_sequence_output, target_pos, target_neg)
                t_rec_loss = self.bpr_loss(t_sequence_output, target_pos, target_neg)
                '''
                n_rec_loss = self.cross_entropy(n_sequence_output, target_pos, target_neg)
                s_rec_loss = self.cross_entropy(s_sequence_output, target_pos, target_neg)
                #t_rec_loss = self.cross_entropy(t_sequence_output, target_pos, target_neg)
                

                # ---------- contrastive learning task -------------#
                cl_losses = []
                n_cl_losses = []
                s_cl_losses = []
                for cl_batch in cl_batches:
                    n_cl_loss, s_cl_loss, t_cl_loss = self._one_pair_contrastive_learning(cl_batch)
                    s_cl_losses.append(s_cl_loss)

                joint_loss = self.args.rec_weight * s_rec_loss
                
                #sim_loss = self.cosine_sim(n_sequence_output, s_sequence_output)
                #sim_loss = 0.5 * ((sim_loss * mask).sum() / (mask.sum() + 1e-9))
                #joint_loss += sim_loss
                #sim_avg_loss += sim_loss
                all_item_emb = self.model.item_embedding.weight
                
                B, seq_len, hidden_dim = s_sequence_output.size()
                # [B*seq_len, hidden_dim]
                n_seq_flat = n_sequence_output.view(-1, hidden_dim)
                s_seq_flat = s_sequence_output.view(-1, hidden_dim)
                
                # [B*seq_len, num_items]
                #n_logits = torch.matmul(n_seq_flat, all_item_emb.transpose(0,1))
                s_logits = torch.matmul(s_seq_flat, all_item_emb.transpose(0,1))
                #s_logits = torch.matmul(s_seq_flat, all_item_emb.transpose(0,1))
                s_log_probs = log_softmax(s_logits, dim=-1)

                # 미리 생성된 equal_probs를 불러온 후 필요한 만큼 잘라줍니다.
                equal_probs = self.equal_probs[:B * seq_len].to(self.device)

                # mask 생성 및 적용
                mask = (input_ids > 0).view(-1).float()
                masked_s_log_probs = s_log_probs[mask.bool()]
                masked_equal_probs = equal_probs[mask.bool()]

                # KL 계산 (reduction='none' 활용)
                kl_each_pos = self.mutual_kl_divergence(masked_s_log_probs, masked_equal_probs)

                # 평균 계산
                ml_loss = kl_each_pos.mean()
                joint_loss += ml_loss
                kl_avg_loss += ml_loss
                
                #sim_loss = self.cosine_sim(n_sequence_output, s_sequence_output)
                #joint_loss += sim_loss
                #sim_avg_loss += sim_loss
                
                param_distance = 0.0
                for p1, p2 in zip(self.model.trm_encoder.parameters(), self.model.trm_encoder2.parameters()):
                    param_distance += torch.sum((p1 - p2)**2)
                joint_loss += param_distance
                sim_avg_loss += param_distance
                '''
                # log softmax
                #n_log_probs = log_softmax(n_logits, dim=-1)
                s_log_probs = log_softmax(s_logits, dim=-1)
                #s_log_probs = torch.sigmoid(s_logits)
                #s_log_probs = log_softmax(s_log_probs, dim=-1)
                equal_probs = torch.full((B * seq_len, all_item_emb.size(0)), 1.0 / all_item_emb.size(0))
                equal_probs = equal_probs.to(self.device)
                ### (B) 패딩 마스크
                # target_pos가 0인 곳은 (학습 시점이 없는) 패딩
                # shape = [B, seq_len] -> 일렬화 -> [B*seq_len]
                mask = (target_pos > 0).view(-1).float()

                ### (C) KL 계산 (reduction='none' 활용)
                kl_each_pos = self.mutual_kl_divergence(s_log_probs, equal_probs)  # [B*seq_len]
                # 패딩 위치 제외 후 평균
                ml_loss = (kl_each_pos * mask).sum() / (mask.sum() + 1e-9)
                joint_loss += ml_loss
                kl_avg_loss += ml_loss
                '''
                for cl_loss in s_cl_losses:
                    joint_loss += self.args.cf_weight * cl_loss
                for param in self.model.trm_encoder2.parameters():
                    param.requires_grad = True
                for param in self.model.trm_encoder.parameters():
                    param.requires_grad = False
                self.model.position_embedding.weight.requires_grad = False
                self.model.position_embedding2.weight.requires_grad = True
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
                "kl_avg_loss": '{:.4f}'.format(kl_avg_loss / len(rec_cf_data_iter)),
                "sim_avg_loss": '{:.4f}'.format(sim_avg_loss / len(rec_cf_data_iter)),
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

                    n_recommend_output, s_recommend_output, t_sequence_output = recommend_output
                    n_recommend_output = n_recommend_output[:,-1,:]
                    # recommendation results

                    rating_pred = self.predict_full(n_recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.n_train_matrix[batch_user_index].toarray() > 0] = 0
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

                    n_recommend_output, s_recommend_output, t_sequence_output = recommend_output
                    s_recommend_output = s_recommend_output[:, -1, :]
                    # recommendation results

                    rating_pred = self.predict_full(s_recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.s_train_matrix[batch_user_index].toarray() > 0] = 0
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
