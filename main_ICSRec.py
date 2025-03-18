import os
import numpy as np
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import RecWithContrastiveLearningDataset,DS

from trainers import ICSRecTrainer
from models import SASRecModel,GRUEncoder
from utils import EarlyStopping, get_user_seqs, check_path, set_seed


def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        print(f"{arg:<30} : {getattr(args, arg):>35}")


def main():
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--data_name", default="Beauty", type=str)
    parser.add_argument("--encoder",default="SAS",type=str) # {"SAS":SASRec,"GRU":GRU4Rec}
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--model_idx", default=2, type=int, help="model idenfier 10, 20, 30...")
    parser.add_argument("--gpu_id", type=str, default="2", help="gpu_id")

    # robustness experiments
    parser.add_argument(
        "--noise_ratio",
        default=0.0,
        type=float,
        help="percentage of negative interactions in a sequence - robustness analysis",
    )

    ## contrastive learning task args
    parser.add_argument(
        "--temperature", default=1.0, type=float, help="softmax temperature (default:  1.0) - not studied."
    )

    parser.add_argument(
        "--intent_num",default=512,type=int,help="the multi intent nums!."
    )

    parser.add_argument(
        "--sim",default='dot',type=str,help="the calculate ways of the similarity."
    )

    # model args
    parser.add_argument("--model_name", default="ICSRec", type=str)
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=2022, type=int)
    # loss weight
    parser.add_argument("--rec_weight", type=float, default=1, help="weight of contrastive learning task")
    parser.add_argument("--lambda_0", type=float, default=0.1, help="weight of coarse-grain intent contrastive learning task")
    parser.add_argument("--beta_0", type=float, default=0.1, help="weight of fine-grain contrastive learning task")

    # ablation experiments
    parser.add_argument("--cl_mode",type=str,default='cf',help="contrastive mode")
    # {'cf':coarse-grain+fine-grain,'c':only coarse-grain,'f':only fine-grain}
    parser.add_argument("--f_neg", action="store_true", help="delete the FNM component (both in cicl and ficl)")

    # learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")

    args = parser.parse_args()
    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + ".txt"
   
    args.n_data_file = args.data_dir + args.data_name + "_n.txt"
    
    args.s_data_file = args.data_dir + args.data_name + "_s.txt"
    args.train_data_file = args.data_dir + args.data_name + "_1.txt"
    args.n_train_data_file = args.data_dir + args.data_name + "_n_1.txt"
    args.s_train_data_file = args.data_dir + args.data_name + "_s_1.txt"
    # construct supervisory signals via DS(·) operation
    if not os.path.exists(args.train_data_file):
        DS(args.data_file,args.train_data_file,args.max_seq_length)
    if not os.path.exists(args.n_train_data_file):
        DS(args.n_data_file,args.n_train_data_file,args.max_seq_length)
    if not os.path.exists(args.s_train_data_file):
        DS(args.s_data_file,args.s_train_data_file,args.max_seq_length)
    
    
    # training data
    a_train_user_id, a_train_user_seq, n_user_id, n_train_user_seq, s_user_id, s_train_user_seq, max_item = get_user_seqs(args, 'train')
    # valid and test data
    n_user_seq, s_user_seq, n_valid_rating_matrix, n_test_rating_matrix, s_valid_rating_matrix, s_test_rating_matrix, max_item = get_user_seqs(args, 'valid')

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    # save model args
    args_str = f"{args.model_name}-{args.encoder}-{args.data_name}-{args.model_idx}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")

    show_args_info(args)

    with open(args.log_file, "a") as f:
        f.write(str(args) + "\n")

    # set item score in train set to `0` in validation
    
    args.n_train_matrix = n_valid_rating_matrix
    args.s_train_matrix = s_valid_rating_matrix
    
     
    args.n_test_matrix = n_test_rating_matrix 
    args.s_test_matrix = s_test_rating_matrix

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

        # training data for node classification
    n_cluster_dataset = RecWithContrastiveLearningDataset(
        args, n_train_user_seq, data_type="train"
    )
    n_cluster_sampler = SequentialSampler(n_cluster_dataset)
    n_cluster_dataloader = DataLoader(n_cluster_dataset, sampler=n_cluster_sampler, batch_size=args.batch_size)
    
    s_cluster_dataset = RecWithContrastiveLearningDataset(
        args, s_train_user_seq, data_type="train"
    )
    s_cluster_sampler = SequentialSampler(s_cluster_dataset)
    s_cluster_dataloader = DataLoader(s_cluster_dataset, sampler=s_cluster_sampler, batch_size=args.batch_size)

    n_train_dataset = RecWithContrastiveLearningDataset(
        args, n_train_user_seq, data_type="train"
    )
    n_train_sampler = RandomSampler(n_train_dataset)
    n_train_dataloader = DataLoader(n_train_dataset, sampler=n_train_sampler, batch_size=args.batch_size)

    n_eval_dataset = RecWithContrastiveLearningDataset(args, n_user_seq, data_type="valid")
    n_eval_sampler = SequentialSampler(n_eval_dataset)
    n_eval_dataloader = DataLoader(n_eval_dataset, sampler=n_eval_sampler, batch_size=args.batch_size)

    n_test_dataset = RecWithContrastiveLearningDataset(args, n_user_seq, data_type="test")
    n_test_sampler = SequentialSampler(n_test_dataset)
    n_test_dataloader = DataLoader(n_test_dataset, sampler=n_test_sampler, batch_size=args.batch_size)
    

    s_train_dataset = RecWithContrastiveLearningDataset(
        args, s_train_user_seq, data_type="train"
    )
    s_train_sampler = RandomSampler(s_train_dataset)
    s_train_dataloader = DataLoader(s_train_dataset, sampler=s_train_sampler, batch_size=args.batch_size)

    s_eval_dataset = RecWithContrastiveLearningDataset(args, s_user_seq, data_type="valid")
    s_eval_sampler = SequentialSampler(s_eval_dataset)
    s_eval_dataloader = DataLoader(s_eval_dataset, sampler=s_eval_sampler, batch_size=args.batch_size)

    s_test_dataset = RecWithContrastiveLearningDataset(args, s_user_seq, data_type="test")
    s_test_sampler = SequentialSampler(s_test_dataset)
    s_test_dataloader = DataLoader(s_test_dataset, sampler=s_test_sampler, batch_size=args.batch_size)


    if args.encoder=="SAS":
        model = SASRecModel(args=args)
    elif args.encoder=="GRU":
        model=GRUEncoder(args=args)
    trainer = ICSRecTrainer(model, n_train_dataloader,  n_eval_dataloader, n_test_dataloader, s_train_dataloader, s_eval_dataloader, s_test_dataloader, n_cluster_dataloader, s_cluster_dataloader, args)

    if args.do_eval:
        
        trainer.load(args.checkpoint_path)

        print(f"Load model from {args.checkpoint_path} for test!")
        scores, result_info = trainer.test(0, full_sort=True)

    else:
        print(f"Train ICSRec")
        best_ndcg = 0.0
        early_num = 0
        best_epoch = 0
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
                best_epoch = epoch
            else:
                early_num += 1 
            print('---------------Change to test_rating_matrix!-------------------')
            
            trainer.args.n_train_matrix = n_test_rating_matrix
            trainer.args.s_train_matrix = s_test_rating_matrix
            recall, ndcg = trainer.test(epoch, full_sort=True)
            #if epoch == 40:
            # best_ndcg = 0.0
            
            trainer.args.n_train_matrix = n_valid_rating_matrix
            trainer.args.s_train_matrix = s_valid_rating_matrix
    
            if early_num == 150:
                break
            print(best_epoch)


main()
