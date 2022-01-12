#!/usr/bin/env python3

import argparse
import random
import os

import numpy as np
import torch
from torch import nn, optim

import learn2learn as l2l
from learn2learn.data.transforms import (
    NWays,
    KShots,
    LoadData,
    RemapLabels,
    ConsecutiveLabels
)

from data_utils import get_data_loaders, get_support_noise_query_indices, preprocess_data_labels
from noise_utils import add_noise
from few_shot_utils import accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ways', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query_num', type=int, default=15)
    parser.add_argument('--test-tasks', type=int, default=10000)
    
    parser.add_argument('--dataset', type=str, default="miniimagenet")
    parser.add_argument('--data-path', type=str, default="data/miniimagenet")
    parser.add_argument('--random-horizontal-flip', action='store_true')
    parser.add_argument('--random-resized-crop', action='store_true')
    parser.add_argument('--color-jitter', action='store_true')
    parser.add_argument('--random-erasing', action='store_true')
    
    parser.add_argument('--noise-type', type=str, default="sym_swap")    
    parser.add_argument('--train-support-label-noise-choices', type=float, nargs='+', default=[0.0])
    parser.add_argument('--test-support-label-noise-list', type=float, nargs='+', default=[0.0])    
    
    parser.add_argument('--num-iterations', type=int, default=60000)
    parser.add_argument('--adaptation-steps-train', type=int, default=5)
    parser.add_argument('--adaptation-steps-test', type=int, default=10)
    parser.add_argument('--fast-lr', type=float, default=0.01)
    parser.add_argument('--meta-lr', type=float, default=0.003)
    parser.add_argument('--meta-batch-size', type=int, default=32)
    
    parser.add_argument('--first-order', action='store_true')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--load-checkpoint-path', type=str, default="")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    
    args.train_way = args.ways
    args.train_shot = args.shot
    args.train_query = args.query_num
    args.test_way = args.ways
    args.test_shot = args.shot
    args.test_query = args.query_num
    
    print(args)
    return args


def fast_adapt(
    batch, learner, loss, adaptation_steps, shot, ways, query_num, device, 
    support_label_noise_choices=[0.0], noise_type="sym_swap", outlier_batch=None
):
    data, labels = batch
    data, labels = preprocess_data_labels(batch, device)
    if outlier_batch:
        outlier_data, _ = preprocess_data_labels(outlier_batch, device)
    else:
        outlier_data = None


    # Separate data into adaptation/evalutation sets
    mask_indices = get_support_noise_query_indices(ways, shot, query_num)
    
    # Add noise
    support_label_noise = np.random.choice(support_label_noise_choices)
    data, labels, noise_positions = add_noise(
        data, labels, mask_indices, ways, support_label_noise, noise_type, outlier_data
    )
    
    adaptation_data = data[mask_indices["support"]]
    evaluation_data = data[mask_indices["query"]]
    adaptation_labels = torch.repeat_interleave(torch.arange(ways), shot).to(device)
    evaluation_labels = torch.repeat_interleave(torch.arange(ways), query_num).to(device)
   
    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy


def main(
):
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cpu')
    if torch.cuda.device_count():
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda:' + str(args.gpu))

    # Create Tasksets using the benchmark interface
    data_loaders = get_data_loaders(args)

    # Create model
    model = l2l.vision.models.MiniImagenetCNN(args.ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=args.first_order)
    opt = optim.Adam(maml.parameters(), args.meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')
    
    # Load checkpoint if available
    if args.load_checkpoint_path != "":
        maml.load_state_dict(
            torch.load(args.load_checkpoint_path, map_location='cuda:'+str(args.gpu))
        )
    
    # Output directory/logging
    output_dirname = os.path.join("/checkpoint/kevinjliang/nfsl/", args.name)
    os.makedirs(output_dirname, exist_ok=True)
    with open(os.path.join(output_dirname, "args.txt"), "w") as f:
        f.write(str(args) + "\n\n")
        f.write(str(model) + "\n")

    # Training
    best_valid_acc = 0
    best_valid_acc_epoch = 0

    for iteration in range(args.num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(args.meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = next(iter(data_loaders["train"]))
            outlier_batch = next(iter(data_loaders["outlier_train"])) if args.noise_type == "outlier" else None
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch,
                learner,
                loss,
                args.adaptation_steps_train,
                args.shot,
                args.ways,
                args.query_num,
                device,
                support_label_noise_choices=args.train_support_label_noise_choices,
                noise_type=args.noise_type,
                outlier_batch=outlier_batch,
            )
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = next(iter(data_loaders["valid"]))
            outlier_batch = next(iter(data_loaders["outlier_test"])) if args.noise_type == "outlier" else None
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch,
                learner,
                loss,
                args.adaptation_steps_test,
                args.shot,
                args.ways,
                args.query_num,
                device,
                support_label_noise_choices=[0.0],
                noise_type=args.noise_type,
                outlier_batch=outlier_batch,
            )
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / args.meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / args.meta_batch_size)
        print('Meta Valid Error', meta_valid_error / args.meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / args.meta_batch_size)
        print('\n')
            
        train_val_log = """
            Iteration: {}\t 
            Meta Train Error: {}\t
            Meta Train Accuracy: {}\t
            Meta Valid Error: {}\t
            Meta Valid Accuracy: {}\t\n
        """.format(
            iteration,
            meta_train_error / args.meta_batch_size,
            meta_train_accuracy / args.meta_batch_size,
            meta_valid_error / args.meta_batch_size,
            meta_valid_accuracy / args.meta_batch_size,
        )
        with open(os.path.join(output_dirname, "train_val_logs.txt"), "a") as f:
            f.write(train_val_log + "\n")
            
        # Save model if best so far
        if meta_valid_accuracy / args.meta_batch_size > best_valid_acc:
            best_valid_acc = meta_valid_accuracy / args.meta_batch_size
            best_valid_acc_epoch = iteration
            checkpoint_path = os.path.join(output_dirname, "checkpoint.pth")
            torch.save(maml.state_dict(), checkpoint_path)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / args.meta_batch_size)
        opt.step()

    for label_noise in args.test_support_label_noise_list:
        meta_test_error = 0.0
        meta_test_accuracy = []
        
        for task in range(args.test_tasks):
            # Compute meta-testing loss
            learner = maml.clone()
            batch = next(iter(data_loaders["test"]))
            outlier_batch = next(iter(data_loaders["outlier_test"])) if args.noise_type == "outlier" else None    
            
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch,
                learner,
                loss,
                args.adaptation_steps_test,
                args.shot,
                args.ways,
                args.query_num,
                device,
                [label_noise],
                args.noise_type,
                outlier_batch,
            )
            meta_test_error += evaluation_error.item()
            meta_test_accuracy.append(evaluation_accuracy.item())
        print('{} {}: Meta Test Error {}'.format(args.noise_type, label_noise, meta_test_error / args.test_tasks))
        print('{} {}: Meta Test Accuracy {} +/- {}'.format(
            args.noise_type, label_noise, np.mean(meta_test_accuracy), 1.96*np.std(meta_test_accuracy)/np.sqrt(args.test_tasks)
        ))
        
        test_log = "{} {}: Meta Test Error: {}\t Meta Test Accuracy: {} +/- {}\n".format(
            args.noise_type, 
            label_noise,
            meta_test_error / args.test_tasks,
            np.mean(meta_test_accuracy),
            1.96*np.std(meta_test_accuracy)/np.sqrt(args.test_tasks)
        )
        with open(os.path.join(output_dirname, "test_logs.txt"), "a") as f:
            f.write(test_log + "\n")



if __name__ == '__main__':
    main()
