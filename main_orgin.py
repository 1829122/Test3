import os
import torch
import shutil
import argparse
import numpy as np

from trainers import *
from datasets import *
from utils import *
from models import OfflineItemSimilarity, SASRec

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def show_args_info(args):
    print(f"--------------------Configure Info:--------------------")
    with open(args.log_file, 'a') as f:
        for arg in vars(args):
            info = f"{arg:<30} : {getattr(args, arg):>35}"
            print(info)
            f.write(info + '\n')


def main():
    parser = argparse.ArgumentParser()

    # system args
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--data_name', type=str, default='Beauty')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--model_idx', type=int, default=35, help="model identifier 1,2,3,4,5,6,7...")
    parser.add_argument("--gpu_id", type=str, default="2", help="gpu_id")
    parser.add_argument("--eval_path", type=str, default='./output', help="checkpoint path for eval")

    # data augmentation args
    parser.add_argument('--noise_ratio', type=float, default=0.0,
                        help="percentage of negative interactions in a sequence - robustness analysis")
    parser.add_argument('--training_data_ratio', type=float, default=1.0,
                        help="percentage of training samples used for training - robustness analysis")
    parser.add_argument('--augment_threshold', type=int, default=12,
                        help="control augmentations on short and long sequences.\
                        default:-1, means all augmentations types are allowed for all sequences.\
                        For sequence length < augment_threshold: TI-Insert, and TI-Substitute methods are allowed \
                        For sequence length > augment_threshold: TI-Crop, TI-Reorder, TI-Substitute, and TI-Mask are allowed.")
    parser.add_argument('--similarity_model_name', type=str, default='ItemCF_IUF',
                        help="Method to generate item similarity score. choices: \
                        Random, ItemCF, ItemCF_IUF(Inverse user frequency), Item2Vec")
    parser.add_argument("--augmentation_warm_up_epochs", default=350, type=int,
                        help="number of epochs to switch from memory-based similarity model to hybrid similarity model.")
    parser.add_argument('--base_augment_type', type=str, default='random',
                        help="default data augmentation types. Chosen from: \
                        mask, crop, reorder, substitute, insert, random, combinatorial_enumerate (for multi-view).")
    parser.add_argument('--augment_type_for_short', type=str, default='SIM',
                        help="data augmentation types for short sequences. Chosen from: \
                        SI, SIM, SIR, SIC, SIMR, SIMC, SIRC, SIMRC.")
    parser.add_argument("--crop_rate", type=float, default=0.7, help="crop ratio for crop operator")
    parser.add_argument("--mask_rate", type=float, default=0.5, help="mask ratio for mask operator")
    parser.add_argument("--reorder_rate", type=float, default=0.2, help="reorder ratio for reorder operator")
    parser.add_argument("--substitute_rate", type=float, default=0.1, help="substitute ratio for substitute operator")
    parser.add_argument("--insert_rate", type=float, default=0.5, help="insert ratio for insert operator")
    parser.add_argument("--max_insert_num_per_pos", type=int, default=1,
                        help="maximum insert items per position for insert operator - not studied")
    parser.add_argument('--insert_mode', default='maximum', type=str,
                        help="minimum or maximum, maximum is the Ti-Insert in paper, minimum is the worst.")
    parser.add_argument('--crop_mode', default='minimum', type=str,
                        help="minimum or maximum, minimum is the Ti-Crop in paper, maximum is the worst.")
    parser.add_argument('--substitute_mode', default='minimum', type=str,
                        help="minimum or maximum, minimum is the Ti-Substitute in paper, maximum is the worst.")
    parser.add_argument('--reorder_mode', default='minimum', type=str,
                        help="minimum or maximum, minimum is the Ti-Reorder in paper, maximum is the worst.")
    parser.add_argument('--mask_mode', default='maximum', type=str, help="minimum, maximum or random")
    parser.add_argument("--var_rank_not_aug_ratio", type=float, default=0.15,
                        help="Using the data ranked by time interval variance, \
                        percentage of training samples (user interaction sequence) will not be augmented")
    parser.add_argument("--model_warm_up_epochs", type=int, default=350,
                        help="number of epochs to train model without contrastive learning. "
                             "If this parameter is 0, contrastive learning will be included from the first epoch")
    parser.add_argument('--not_aug_data_mode', default='original', type=str, help="original or zero\
                        original: for non-augmented users, return the original sequence \
                        zero: for non-augmented users, return the zero sequence [0,0,,...,0,0,0]")

    # contrastive learning task args
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='softmax temperature (default:  1.0) - not studied.')
    parser.add_argument('--n_views', type=int, default=2, metavar='N',
                        help='Number of augmented data for each sequence - not studied.')

    # model args
    parser.add_argument("--model_name", default='TiCoSeRec', type=str)
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="the input and output hidden size (embedding size)")
    parser.add_argument("--n_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--n_heads', type=int, default=2, help="number of heads")
    parser.add_argument('--inner_size', type=int, default=256, help='the dimensionality in feed-forward layer')
    parser.add_argument('--hidden_act', type=str, default="gelu")
    parser.add_argument("--attn_dropout_prob", type=float, default=0.2, help="attention dropout probability")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout probability")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', type=int, default=50)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=350, help="number of epochs")
    parser.add_argument("--patience", type=int, default=150, help="early stopping patience")
    parser.add_argument("--test_frequency", type=int, default=5, help="test frequency")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--cf_weight", type=float, default=0.1, help="weight of contrastive learning task")
    parser.add_argument("--rec_weight", type=float, default=1.0, help="weight of contrastive learning task")

    # learning related
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    

    args.n_data_file = os.path.join(args.data_dir, args.data_name, args.data_name + '_n_item.txt')
    args.n_time_file = os.path.join(args.data_dir, args.data_name, args.data_name + '_n_time.txt')
    args.s_data_file = os.path.join(args.data_dir, args.data_name, args.data_name + '_s_item.txt')
    args.s_time_file = os.path.join(args.data_dir, args.data_name, args.data_name + '_s_time.txt')
    args.data_file = os.path.join(args.data_dir, args.data_name, args.data_name + '_item.txt')
    args.time_file = os.path.join(args.data_dir, args.data_name, args.data_name + '_time.txt')

    a_user_seq, a_time_seq,n_user_seq, n_time_seq,s_user_seq, s_time_seq, max_item, valid_rating_matrix, test_rating_matrix,  not_aug_users, n_valid_rating_matrix, n_test_rating_matrix, s_valid_rating_matrix, s_test_rating_matrix = get_user_seqs(args)
    
    args.item_size = max_item + 2
    args.mask_id = max_item + 1

    # save model and args
    args_str = f'{args.model_name}-{args.data_name}-{args.model_idx}'
    args.checkpoint_path = os.path.join(args.output_dir, args_str)
    if os.path.exists(args.checkpoint_path):
        shutil.rmtree(args.checkpoint_path, ignore_errors=True)
    os.mkdir(args.checkpoint_path)
    args.log_file = os.path.join(args.checkpoint_path, args_str + '.txt')
    args.test_log_file = os.path.join(args.checkpoint_path, args_str + '-test.txt')

    show_args_info(args)

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix
    args.n_train_matrix = n_valid_rating_matrix
    args.s_train_matrix = s_valid_rating_matrix
    # -----------   pre-computation for item similarity   ------------ #
    args.similarity_model_path = os.path.join(args.data_dir,
                                              args.data_name + '_' + args.similarity_model_name + '_similarity.pkl')
    offline_similarity_model = OfflineItemSimilarity(data_file=args.data_file,
                                                     similarity_path=args.similarity_model_path,
                                                     model_name=args.similarity_model_name, dataset_name=args.data_name)
    args.offline_similarity_model = offline_similarity_model


    # training data for node classification
    n_train_dataset = DataSets(args, n_user_seq[:int(len(n_user_seq) * args.training_data_ratio)],
                                                      n_time_seq, not_aug_users=not_aug_users, data_type='train')
    n_train_sampler = RandomSampler(n_train_dataset)
    n_train_dataloader = DataLoader(n_train_dataset, sampler=n_train_sampler, batch_size=args.batch_size)

    n_eval_dataset = DataSets(args, n_user_seq, n_time_seq,not_aug_users=not_aug_users ,data_type='valid')
    n_eval_sampler = SequentialSampler(n_eval_dataset)
    n_eval_dataloader = DataLoader(n_eval_dataset, sampler=n_eval_sampler, batch_size=args.batch_size)

    n_test_dataset = DataSets(args, n_user_seq, n_time_seq, data_type='test')
    n_test_sampler = SequentialSampler(n_test_dataset)
    n_test_dataloader = DataLoader(n_test_dataset, sampler=n_test_sampler, batch_size=args.batch_size)
    
    s_train_dataset = DataSets(args, s_user_seq[:int(len(s_user_seq) * args.training_data_ratio)],
                                                      s_time_seq, not_aug_users=not_aug_users, data_type='train')
    s_train_sampler = RandomSampler(s_train_dataset)
    s_train_dataloader = DataLoader(s_train_dataset, sampler=s_train_sampler, batch_size=args.batch_size)

    s_eval_dataset = DataSets(args, s_user_seq, s_time_seq, data_type='valid')
    s_eval_sampler = SequentialSampler(s_eval_dataset)
    s_eval_dataloader = DataLoader(s_eval_dataset, sampler=s_eval_sampler, batch_size=args.batch_size)

    s_test_dataset = DataSets(args, s_user_seq, s_time_seq, data_type='test')
    s_test_sampler = SequentialSampler(s_test_dataset)
    s_test_dataloader = DataLoader(s_test_dataset, sampler=s_test_sampler, batch_size=args.batch_size)

    model = SASRec(args=args)

    trainer = CoSeRecTrainer(model, n_train_dataloader, n_eval_dataloader,
                             n_test_dataloader, s_train_dataloader, s_eval_dataloader,
                             s_test_dataloader, args)

    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.eval_path)
        print(f'Load model from {args.eval_path} for test!')
        scores, result_info = trainer.test(0, full_sort=True)

    else:
        print(f'Train TiCoSeRec')
        #early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        best_ndcg = 0.0
        early_num = 0
        for epoch in range(args.epochs):
            
            trainer.train(epoch)
            # evaluate on NDCG@20
            #recall, ndcg = trainer.valid(epoch, full_sort=True)
         
            #early_stopping(np.array(scores[-1:]), trainer.model)
            #if early_stopping.early_stop:
                #print("Early stopping")
                #break
            
            
            
            valid_recall, valid_ndcg = trainer.valid(epoch, full_sort=True)
            if valid_ndcg[1] > best_ndcg:
                early_num = 0
                best_ndcg = valid_ndcg[1]
                save_path = os.path.join(args.checkpoint_path, 'epoch-' + str(epoch) + '.pt')
                torch.save(trainer.model.state_dict(), save_path)
            else:
                early_num += 1 
            print('---------------Change to test_rating_matrix!-------------------')
            trainer.args.train_matrix = test_rating_matrix
            trainer.args.n_train_matrix = n_test_rating_matrix
            trainer.args.s_train_matrix = s_test_rating_matrix
            recall, ndcg = trainer.test(epoch, full_sort=True)
            #if epoch == 40:
            # best_ndcg = 0.0
            
            
            trainer.args.train_matrix = valid_rating_matrix
            trainer.args.n_train_matrix = n_valid_rating_matrix
            trainer.args.s_train_matrix = s_valid_rating_matrix
    
            if early_num == 150:
                break

    print('Finish training')


main()