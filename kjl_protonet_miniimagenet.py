#!/usr/bin/env python3

import argparse
import datetime
import numpy as np
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from visdom import Visdom

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
from warmup_scheduler import GradualWarmupScheduler

from data_utils import get_data_loaders
from models import ProtoConvnet, ProtoConvTransformer
from protonet_utils import MetaTrainer


if __name__ == '__main__':
    train_start = time.time()
    
    parser = argparse.ArgumentParser()
    # Few-shot setting
    parser.add_argument('--train-shot', type=int, default=5)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--train-query', type=int, default=15)
    parser.add_argument('--test-shot', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--test-query', type=int, default=15)
    parser.add_argument('--test-tasks', type=int, default=10000)
    # Method
    parser.add_argument('--agg-method', type=str, default="mean", 
                        help='Aggregation method. Choices: mean, median, cosine_[T], euclidean_[T], abs_[T]')
    parser.add_argument('--comp-method', type=str, default="proto", 
                        help='Comparison method. Choices: proto, nearest_[k], match, lin_cls, rnnp_soft, rnnp_hard')
    parser.add_argument('--dist-metric', type=str, default="euclidean", 
                        help='Distance metric. Choices: euclidean, cosine')
    parser.add_argument('--logit-temperature', type=float, default=1.0)
    parser.add_argument('--rnnp-alpha', type=float, default=0.8)
    parser.add_argument('--rnnp-beta', type=int, default=4)
    parser.add_argument('--rnnp-iters', type=int, default=3)    
    # Dataset
    parser.add_argument('--dataset', type=str, default="miniimagenet")
    parser.add_argument('--data-path', type=str, default="data/miniimagenet")
    parser.add_argument('--random-horizontal-flip', action='store_true')
    parser.add_argument('--random-resized-crop', action='store_true')
    parser.add_argument('--color-jitter', action='store_true')
    parser.add_argument('--random-erasing', action='store_true')
    # Noise args
    parser.add_argument('--noise-type', type=str, default="sym_swap")    
    parser.add_argument('--train-support-label-noise-choices', type=float, nargs='+', default=[0.0])
    parser.add_argument('--train-query-label-noise', type=float, default=0.0)
    parser.add_argument('--test-support-label-noise-list', type=float, nargs='+', default=[0.0])
    parser.add_argument('--binary-outlier-loss-weight', type=float, default=0.0)
    parser.add_argument('--clean-proto-loss-weight', type=float, default=0.0)
    # Conv/Transformer args
    parser.add_argument('--conv-model', type=str, default="conv4")
    parser.add_argument('--freeze-conv', action='store_true')
    parser.add_argument('--conv-proj-dim', type=int, default=None)
    parser.add_argument('--transformer-layers', type=int, default=0)
    parser.add_argument('--trans-d-model', type=int, default=128)    
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--transformer-metric', type=str, default="dot_prod")
    parser.add_argument('--ortho-proj', action='store_true')
    parser.add_argument('--ortho-proj-residual', action='store_true')
    parser.add_argument('--cls-type', type=str, default="cls_learn")
    parser.add_argument('--pos-type', type=str, default="pos_learn")
    parser.add_argument('--output-from-cls', action='store_true')
    parser.add_argument('--embed-proto-loss-weight', type=float, default=0.0)
    parser.add_argument('--embed-proto-loss-weight-gamma', type=float, default=0.8)
    parser.add_argument('--embed-proto-loss-weight-step', type=int, default=100)
    parser.add_argument('--res12-load-checkpoint-path', type=str, default="./initialization/miniimagenet/sskd_gen0_ours.pt")
    # Learning rate args
    parser.add_argument('--optimizer', type=str, default="adamw")
    parser.add_argument('--scheduler', type=str, default="steplr")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max-epoch', type=int, default=250)
    parser.add_argument('--step-rate', type=int, default=25)
    parser.add_argument('--step-gamma', type=float, default=0.5)
    parser.add_argument('--warm-up-epochs', type=int, default=0)
    # Administrative
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--load-checkpoint-path', type=str, default="")
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    print(args)

    # Random seeds and devices
    np.random.seed(args.seed) 
    device = torch.device('cpu')
    torch.manual_seed(args.seed)
    if torch.cuda.device_count():
        print("Using gpu")
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda:' + str(args.gpu))
    
    # Visdom
    viz = Visdom(env=args.name.replace("/", "__"))
        
    # Model Initialization
    ways = {"train": args.train_way, "test": args.test_way}
    shot = {"train": args.train_shot, "test": args.test_shot}
    if args.transformer_layers > 0:
        model = ProtoConvTransformer(
            ways, shot, device, conv_model=args.conv_model, conv_proj_dim=args.conv_proj_dim,
            trans_layers=args.transformer_layers, trans_d_model=args.trans_d_model, nhead=args.nhead, 
            ortho_proj=args.ortho_proj, ortho_proj_residual=args.ortho_proj_residual, 
            cls_type=args.cls_type, pos_type=args.pos_type, agg_method=args.agg_method, 
            transformer_metric=args.transformer_metric, output_from_cls=args.output_from_cls, 
            binary_outlier_detection=(args.binary_outlier_loss_weight > 0.0),
            res12_load_checkpoint_path=args.res12_load_checkpoint_path
        )
    else:
        model = ProtoConvnet(
            ways, shot, model=args.conv_model, conv_proj_dim=args.conv_proj_dim, 
            agg_method=args.agg_method, binary_outlier_detection=(args.binary_outlier_loss_weight > 0.0), 
            res12_load_checkpoint_path=args.res12_load_checkpoint_path
        )
    model.to(device)
    print(model)
    
    # Freeze convolutional parameters
    if args.freeze_conv:
        for p in model.convnet.parameters():
            p.requires_grad = False
    
    # Set up MetaTrainer
    meta_trainer = MetaTrainer(model, device, args)
    
    # Optimizer and learning schedule
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    elif args.optimizer == "nesterov":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005
        )
    else:
        raise NotImplementedError
    
    if args.scheduler == "steplr":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_rate, gamma=args.step_gamma
        )
    elif args.scheduler == "cyclic_tri":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=0.000001, max_lr=0.0005, step_size_up=150, mode="triangular", cycle_momentum=False
        )
    elif args.scheduler == "cyclic_tri2":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=0.000001, max_lr=0.0005, step_size_up=150, mode="triangular2", cycle_momentum=False
        )
    elif args.scheduler == "cyclic_exp":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=0.000001, max_lr=0.0005, step_size_up=150, mode="exp_range", gamma=0.998, cycle_momentum=False
        )
    else:
        raise NotImplementedError
    if args.warm_up_epochs > 0:
        lr_scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=args.warm_up_epochs, after_scheduler=lr_scheduler
        )

    # Data loading
    data_loaders = get_data_loaders(args)

    # Output directory/logging
    output_dirname = os.path.join("/checkpoint/kevinjliang/nfsl/", args.name)
    os.makedirs(output_dirname, exist_ok=True)
    with open(os.path.join(output_dirname, "args.txt"), "w") as f:
        f.write(str(args) + "\n\n")
        f.write(str(model) + "\n")
        
    # Initialize visdom windows
    acc_win = viz.line(
        X=np.column_stack((0,0)), Y=np.column_stack((0,0)),
        opts={'title': "Accuracy", 'xlabel': "Epoch", 'ylabel': "Accuracy", 'show_legend': True},
    )
    loss_win = viz.line(
        X=np.column_stack((0,0)), Y=np.column_stack((0,0)),
        opts={'title': "Loss", 'xlabel': "Epoch", 'ylabel': "Loss", 'ytickmax': 2, 'show_legend': True},
    )
    lr_win = viz.line(
        X=np.array([0]), Y=np.array([0]),
        opts={'title': "Learning Rate", 'xlabel': "Epoch", 'ylabel': "Learning Rate",},
    )
    if args.embed_proto_loss_weight > 0:
        embed_proto_loss_win = viz.line(
            X=np.column_stack((0,0)), Y=np.column_stack((0,0)),
            opts={'title': "Embed Proto Loss", 'xlabel': "Epoch", 'ylabel': "Loss", 'ytickmax': 2, 'show_legend': True},
        )       
#         embed_proto_loss_weight_win = viz.line(
#             X=np.array([args.embed_proto_loss_weight]), Y=np.array([0]),
#             opts={'title': "Embed Proto Loss Weight", 'xlabel': "Epoch", 'ylabel': "Weight",},
#         )
    if args.binary_outlier_loss_weight > 0.0:
        outlier_loss_win = viz.line(
            X=np.column_stack((0,0)), Y=np.column_stack((0,0)),
            opts={'title': "Outlier Loss", 'xlabel': "Epoch", 'ylabel': "Loss", 'ytickmax': 2, 'show_legend': True},
        ) 
        outlier_acc_win = viz.line(
            X=np.column_stack((0,0)), Y=np.column_stack((0,0)),
            opts={'title': "Outlier Accuracy", 'xlabel': "Epoch", 'ylabel': "Accuracy", 'show_legend': True},
        )
    if args.clean_proto_loss_weight > 0.0:
        clean_proto_loss_win = viz.line(
            X=np.column_stack((0,0)), Y=np.column_stack((0,0)),
            opts={'title': "Clean Proto Loss", 'xlabel': "Epoch", 'ylabel': "Loss", 'ytickmax': 2, 'show_legend': True},
        ) 
        

    # Load model checkpoint if specified; can be partial    
    if args.load_checkpoint_path != "":
        model.load_state_dict(
            torch.load(args.load_checkpoint_path, map_location='cuda:'+str(args.gpu)), strict=False
        )
    # Freeze convolutional parameters
    if args.freeze_conv:
        for p in model.convnet.parameters():
            p.requires_grad = False

    # Training 
    best_valid_acc = 0
    best_valid_acc_epoch = 0
    
    for epoch in range(1, args.max_epoch + 1):
        # Train step
        model.train()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        
        # Decaying embed_proto_loss_weight (TODO: move this into meta-trainer)
#         embed_proto_loss_weight_decay = args.embed_proto_loss_weight_gamma**(
#             epoch//args.embed_proto_loss_weight_step
#         )
#         embed_proto_loss_weight = args.embed_proto_loss_weight*embed_proto_loss_weight_decay 

        for i in range(100):
            batch = next(iter(data_loaders["train"]))
            outlier_batch = next(iter(data_loaders["outlier_train"])) if args.noise_type == "outlier" else None

            train_losses, train_accs = meta_trainer.proto_loss_acc(
                batch, outlier_batch, args.train_support_label_noise_choices
            )
            
            loss = torch.sum(torch.stack(list(train_losses.values())))

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += train_accs["accuracy"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        # Train Logging 
        train_acc = n_acc.cpu().numpy()/loss_ctr
        train_loss = n_loss/loss_ctr
        train_log = 'epoch {}, train, loss={:.4f} acc={:.4f}'.format(
            epoch, train_loss, train_acc
        )
        print(train_log)
        with open(os.path.join(output_dirname, "train_val_logs.txt"), "a") as f:
            f.write(train_log + "\n")

        # Validation step
        model.eval()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        for i, batch in enumerate(data_loaders["valid"]):
            outlier_batch = next(iter(data_loaders["outlier_test"])) if args.noise_type == "outlier" else None
            
            valid_losses, valid_accs = meta_trainer.proto_loss_acc(
                batch, outlier_batch, support_label_noise_choices=[0.0]
            )
            
            loss = torch.sum(torch.stack(list(valid_losses.values())))

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += valid_accs["accuracy"]

        # Validation logging
        valid_acc = n_acc.cpu().numpy()/loss_ctr
        valid_loss = n_loss/loss_ctr
        val_log = 'epoch {}, val, loss={:.4f} acc={:.4f}'.format(
            epoch, valid_loss, valid_acc
        )
        print(val_log)
        with open(os.path.join(output_dirname, "train_val_logs.txt"), "a") as f:
            f.write(val_log + "\n")
            
        # Save model if best so far
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_valid_acc_epoch = epoch
            checkpoint_path = os.path.join(output_dirname, "checkpoint.pth")
            torch.save(model.state_dict(), checkpoint_path)
        
        # Visdom logging of train and validation
        viz.line(
            X=np.column_stack((epoch,epoch)), Y=np.column_stack((train_acc, valid_acc)),
            win=acc_win, update='append'
        )
        viz.line(
            X=np.column_stack((epoch,epoch)), Y=np.column_stack((train_loss, valid_loss)),
            win=loss_win, update='append'
        )
        viz.line(
            X=np.array([epoch]), Y=np.array([lr_scheduler.get_last_lr()]), 
            win=lr_win, update='append'
        )
        if args.embed_proto_loss_weight > 0:
            viz.line(
                X=np.column_stack((epoch,epoch)), 
                Y=np.column_stack(
                    (train_losses["conv_embed_proto"].detach().cpu(), valid_losses["conv_embed_proto"].detach().cpu())
                ),
                win=embed_proto_loss_win, update='append'
            )
#             viz.line(
#                 X=np.array([epoch]), Y=np.array([embed_proto_loss_weight]), 
#                 win=embed_proto_loss_weight_win, update='append'
#             )
        if args.binary_outlier_loss_weight > 0.0:
            viz.line(
                X=np.column_stack((epoch,epoch)), 
                Y=np.column_stack(
                    (train_losses["binary_outlier"].detach().cpu(), valid_losses["binary_outlier"].detach().cpu())
                ),
                win=outlier_loss_win, update='append'
            )            
            viz.line(
                X=np.column_stack((epoch,epoch)), 
                Y=np.column_stack(
                    (train_accs["binary_outlier"].detach().cpu(), valid_accs["binary_outlier"].detach().cpu())
                ),
                win=outlier_acc_win, update='append'
            )
        if args.clean_proto_loss_weight > 0.0:
            viz.line(
                X=np.column_stack((epoch,epoch)), 
                Y=np.column_stack(
                    (train_losses["dist_to_clean_protos"].detach().cpu(), valid_losses["dist_to_clean_protos"].detach().cpu())
                ),
                win=clean_proto_loss_win, update='append'
            )

#     # Save model
#     checkpoint_path = os.path.join(output_dirname, "final_checkpoint.pth")
#     torch.save(model.state_dict(), checkpoint_path)

    # Write total training time
    train_end = time.time()
    train_time_log = "Training time: {}".format(
        str(datetime.timedelta(seconds=train_end - train_start))
    )
    print(train_time_log)
    with open(os.path.join(output_dirname, "train_val_logs.txt"), "a") as f:
        f.write(train_time_log + "\n")
    
    ## Test
    # Load best model
    test_start = time.time()
    if args.max_epoch > 0:
        model.load_state_dict(torch.load(checkpoint_path))
        with open(os.path.join(output_dirname, "test_logs.txt"), "w") as f:
            f.write("Best epoch: {} ({})\n".format(best_valid_acc_epoch, best_valid_acc))
  
    # Fix random seeds again
    np.random.seed(args.seed) 
    torch.manual_seed(args.seed)
    if torch.cuda.device_count():
        torch.cuda.manual_seed(args.seed)

    model.eval()
    for label_noise in args.test_support_label_noise_list:
        test_accs = []
        for i, batch in enumerate(data_loaders["test"], 1):
            outlier_batch = next(iter(data_loaders["outlier_test"])) if args.noise_type == "outlier" else None
            
            if args.comp_method.startswith("nearest"):
                _, accs = meta_trainer.nearest_neighbor(
                    batch, outlier_batch, support_label_noise_choices=[label_noise]
                )
            else:
                _, accs = meta_trainer.proto_loss_acc(
                    batch, outlier_batch, support_label_noise_choices=[label_noise]
                )
            test_accs.append(accs["accuracy"].cpu())
            # +/- 95% CI (1.96 * STD/sqrt(n))
            print(
                'batch {}: {:.2f}+/-{:.2f} ({:.2f})'.format(
                    i, np.mean(test_accs) * 100, 
                    1.96*np.std(test_accs)/np.sqrt(i+1) * 100, 
                    accs["accuracy"] * 100
                )
            )

        test_log = 'Noise {} {}: {:.2f}+/-{:.2f}'.format(
            args.noise_type, label_noise, np.mean(test_accs) * 100, 
            1.96*np.std(test_accs)/np.sqrt(args.test_tasks) * 100,
        )
        with open(os.path.join(output_dirname, "test_logs.txt"), "a") as f:
            f.write(test_log + "\n")

    
    # Write total test time
    test_end = time.time()
    test_time_log = "Test time: {}".format(
        str(datetime.timedelta(seconds=test_end - test_start))
    )
    print(test_time_log)
    with open(os.path.join(output_dirname, "test_logs.txt"), "a") as f:
        f.write(test_time_log + "\n")
