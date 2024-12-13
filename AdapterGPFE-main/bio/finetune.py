import argparse
from loader import BioDataset
from dataloader import DataLoaderFinetune
from splitters import random_split, species_split
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import GNN_graphpred
from adapterGPFE import AdapterGPFE_graphpred
from sklearn.metrics import roc_auc_score
import os
import logging
import statistics
import gpfe as Prompt

criterion = nn.BCEWithLogitsLoss()


def train(args, model, device, loader, optimizer,prompt):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch,prompt)

        # pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, prompt)


        y = batch.go_target_downstream.view(pred.shape).to(torch.float64)

        optimizer.zero_grad()
        loss = criterion(pred.double(), y)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader,prompt):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch,prompt)
            # pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, prompt)


        y_true.append(batch.go_target_downstream.view(pred.shape).detach().cpu())
        y_scores.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_scores = torch.cat(y_scores, dim=0).numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            roc_list.append(roc_auc_score(y_true[:, i], y_scores[:, i]))
        else:
            roc_list.append(np.nan)

    return sum(roc_list) / len(roc_list)


def main(runseed):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=2,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--model_file', type=str, default='model_gin/masking.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='', help='output filename')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting dataset.")
    parser.add_argument('--runseed', type=int, default=runseed, help="Seed for running experiments.")
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--split', type=str, default="species", help='Random or species split')
    parser.add_argument('--log', type=str, default='AdapterGPFE_masking')





    parser.add_argument('--p_num', type=int, default=3, help='The number of independent basis')
    parser.add_argument('--tuning_type', type=str, default="gpfe", help='')

    parser.add_argument('--max_bottleneck_dim', type=int, default=16, help='Maximum bottleneck dimension')
    parser.add_argument('--min_bottleneck_dim', type=int, default=3, help='Minimum bottleneck dimension')









    args = parser.parse_args()

    logging.basicConfig(format='%(message)s', level=logging.INFO, filename='log/{}.log'.format(args.log), filemode='a')

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    root_supervised = 'dataset/supervised'

    dataset = BioDataset(root_supervised, data_type='supervised')

    print(dataset)

    if args.split == "random":
        print("random splitting")
        train_dataset, valid_dataset, test_dataset = random_split(dataset, seed=args.seed)
    elif args.split == "species":
        trainval_dataset, test_dataset = species_split(dataset)
        train_dataset, valid_dataset, _ = random_split(trainval_dataset, seed=args.seed, frac_train=0.85,
                                                       frac_valid=0.15, frac_test=0)
        test_dataset_broad, test_dataset_none, _ = random_split(test_dataset, seed=args.seed, frac_train=0.5,
                                                                frac_valid=0.5, frac_test=0)
        print("species splitting")
    else:
        raise ValueError("Unknown split name.")

    train_loader = DataLoaderFinetune(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)
    val_loader = DataLoaderFinetune(valid_dataset, batch_size=10 * args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)

    if args.split == "random":
        test_loader =  DataLoaderFinetune(test_dataset, batch_size=10 * args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    else:
        ### for species splitting
        test_easy_loader = DataLoaderFinetune(test_dataset_broad, batch_size=10 * args.batch_size, shuffle=False,
                                              num_workers=args.num_workers)
        # test_hard_loader = DataLoaderFinetune(test_dataset_none, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)

    num_tasks = len(dataset[0].go_target_downstream)

    print(train_dataset[0])

    # set up model
    model = AdapterGPFE_graphpred(args,args.num_layer, args.emb_dim, num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio,
                                 graph_pooling=args.graph_pooling, gnn_type=args.gnn_type, max_bottleneck_dim=args.max_bottleneck_dim,
                                 min_bottleneck_dim=args.min_bottleneck_dim)




    if not args.model_file == "":
        model.from_pretrained(args.model_file)

    model.to(device)








    if args.tuning_type == 'gpfe':
        prompt = Prompt.SimplePrompt(args.emb_dim, args.p_num).to(device)







    # baseline
    if type(model) is GNN_graphpred:
        model_param_group = []
        model_param_group.append({"params": model.gnn.parameters()})
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr})

    # gpfe
    if type(model) is AdapterGPFE_graphpred:
        model_param_group = []

        model_param_group.append({"params": prompt.parameters()})



        model_param_group.append({"params": model.gnn.prompts.parameters(), "lr": args.lr})
        model_param_group.append({"params": model.gnn.gating_parameter, "lr": args.lr})
        for name, p in model.gnn.named_parameters():
            if name.startswith('batch_norms'):
                model_param_group.append({"params": p})
            if 'mlp' in name and name.endswith('bias'):
                model_param_group.append({"params": p})
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr})

    # set up optimizer
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)



    best_val = 0
    select_acc_easy = 0

    if not args.filename == "":
        if os.path.exists(args.filename):
            print("removed existing file!!")
            os.remove(args.filename)

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train(args, model, device, train_loader, optimizer,prompt)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader,prompt)
        else:
            train_acc = 0
            print("ommitting training evaluation")

        val_acc = eval(args, model, device, val_loader,prompt)
        test_acc_easy = eval(args, model, device, test_easy_loader,prompt)


        print("train: %f val: %f test: %f" % (train_acc, val_acc, test_acc_easy))




        if val_acc > best_val:
            # logging.info('{:.2f} {:.2f} ---'.format(100 * val_acc, 100 * test_acc_easy))
            best_val = val_acc
            select_acc_easy = test_acc_easy
        # else:
            # logging.info('{:.2f} {:.2f}'.format(100 * val_acc, 100 * test_acc_easy))

    return select_acc_easy


if __name__ == "__main__":
    total_acc = []
    repeat = 1
    for runseed in range(repeat):
        acc = main(runseed)
        total_acc.append(acc)
    logging.info(
        'easy: {:.2f}±{:.2f}'.format(100 * sum(total_acc) / len(total_acc), 100 * statistics.pstdev(total_acc)))
