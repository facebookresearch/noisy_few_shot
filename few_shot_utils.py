"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import preprocess_data_labels, get_support_noise_query_indices
from noise_utils import add_noise


def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -(
        (a.unsqueeze(1).expand(n, m, -1) - b.unsqueeze(0).expand(n, m, -1)) ** 2
    ).sum(dim=2)
    return logits


def pairwise_cosine_logits(a, b, epsilon=1e-6):
    # Normalize all embeddings to unit vectors
    norm_a = a / (torch.norm(a, dim=1, keepdim=True) + epsilon)
    norm_b = b / (torch.norm(b, dim=1, keepdim=True) + epsilon)

    # Calculate cosine angle between all support samples in each class
    cos = norm_a @ norm_b.T

    return cos


def accuracy(predictions, targets, binary=False):
    if binary:
        predictions = (predictions > 0.5).float().view(targets.shape)
    else:
        predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


#####################
# Prototypes
#####################


def gen_prototypes(embeddings, ways, shots, agg_method="mean"):
    assert (
        embeddings.size(0) == ways * shots
    ), "# of embeddings ({}) doesn't match ways ({}) and shots ({})".format(
        embeddings.size(0), ways, shots
    )

    embeddings = embeddings.reshape(ways, shots, -1)
    mean_embeddings = embeddings.mean(dim=1)

    if agg_method == "mean":
        return mean_embeddings

    elif agg_method == "median":
        # Init median as mean
        median_embeddings = torch.unsqueeze(mean_embeddings, dim=1)
        c = 0.5
        for i in range(5):
            errors = median_embeddings - embeddings
            # Poor man's Newton's method
            denom = torch.sqrt(torch.sum(errors ** 2, axis=2, keepdims=True) + c ** 2)
            dw = -torch.sum(errors / denom, axis=1, keepdims=True) / torch.sum(
                1.0 / denom, axis=1, keepdims=True
            )
            median_embeddings += dw
        return torch.squeeze(median_embeddings, dim=1)

    elif (
        agg_method.startswith("cosine")
        or agg_method.startswith("euclidean")
        or agg_method.startswith("abs")
    ):
        epsilon = 1e-6

        if agg_method.startswith("cosine"):
            # Normalize all embeddings to unit vectors
            norm_embeddings = embeddings / (
                torch.norm(embeddings, dim=2, keepdim=True) + epsilon
            )
            # Calculate cosine angle between all support samples in each class: ways x shots x shots
            # Make negative, as higher cosine angle means greater correlation
            cos = torch.bmm(norm_embeddings, norm_embeddings.permute(0, 2, 1))
            attn = (torch.sum(cos, dim=1) - 1) / (shots - 1)
        elif agg_method.startswith("euclidean"):
            # dist: ways x shots x shots
            dist = (
                (embeddings.unsqueeze(dim=2) - embeddings.unsqueeze(dim=1)) ** 2
            ).sum(dim=-1)
            attn = -torch.sum(dist, dim=1) / (shots - 1)
        elif agg_method.startswith("abs"):
            # dist: ways x shots x shots
            dist = (
                torch.abs(embeddings.unsqueeze(dim=2) - embeddings.unsqueeze(dim=1))
            ).sum(dim=-1)
            attn = -torch.sum(dist, dim=1) / (shots - 1)

        # Parse softmax temperature (default=1)
        T = float(agg_method.split("_")[-1]) if "_" in agg_method else 1
        weights = F.softmax(attn / T, dim=1).unsqueeze(dim=2)
        weighted_embeddings = embeddings * weights
        return weighted_embeddings.sum(dim=1)

    else:
        raise NotImplementedError


def gen_subset_prototypes(
    support_embeddings, shot, ways, subset_proportion=0.4, num_ensembles=10
):
    support_prototypes_sets = []
    num_support_samples = int(subset_proportion * shot)
    mask_array = [True] * num_support_samples + [False] * (shot - num_support_samples)

    for i in range(num_ensembles):
        subsample_indices_mask_i = np.concatenate(
            [np.random.permutation(mask_array) for _ in range(ways)]
        )
        sampled_support_embeddings = support_embeddings[subsample_indices_mask_i]
        support_prototypes_i = sampled_support_embeddings.reshape(
            ways, num_support_samples, -1
        ).mean(dim=1)
        support_prototypes_sets.append(support_prototypes_i)

    return support_prototypes_sets


def subset_prototype_predictions(subset_protos, query_embeddings, ways):
    # Make prediction with all sub-prototypes
    predictions = []
    for subset_protos_i in subset_protos:
        logits_i = pairwise_distances_logits(query_embeddings, subset_protos_i)
        predictions.append(torch.argmax(logits_i, dim=1))
    ensemble_predictions = torch.vstack(predictions)

    # Pick most common prediction as final prediction
    final_predictions = []
    for preds_i in ensemble_predictions.T:
        # Count up predictions of ensemble per class
        pred_counts_i = torch.bincount(preds_i, minlength=ways)
        # Find the class(es) with the most predictions
        (max_preds_i,) = torch.where(pred_counts_i == pred_counts_i.max())
        # Randomly select one of the classes with max predictions (if tie)
        random_index = np.random.randint(len(max_preds_i))
        final_predictions.append(max_preds_i[random_index])

    return torch.hstack(final_predictions)


#####################
# MetaTrainer
#####################


class MetaTrainer:
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args

        if args.dist_metric == "euclidean":
            self.metric = pairwise_distances_logits
        elif args.dist_metric == "cosine":
            self.metric = pairwise_cosine_logits
        else:
            raise NotImplementedError

    def prepare_data(
        self,
        batch,
        ways,
        shot,
        query_num,
        support_label_noise_choices,
        outlier_batch=None,
    ):
        # Format data and place on device
        data, labels = preprocess_data_labels(batch, self.device)
        if outlier_batch:
            outlier_data, _ = preprocess_data_labels(outlier_batch, self.device)
        else:
            outlier_data = None

        # Separate query and support
        mask_indices = get_support_noise_query_indices(ways, shot, query_num)

        # Add noise
        support_label_noise = np.random.choice(support_label_noise_choices)
        noise_positions = np.zeros(data.shape[0])
        if support_label_noise > 0:
            data, labels, noise_positions = add_noise(
                data,
                labels,
                mask_indices,
                ways,
                support_label_noise,
                self.args.noise_type,
                outlier_data,
            )

        # Remove noise batch data for swapping, so conv doesn't need to process
        data = data[~mask_indices["noise"]]
        labels = labels[~mask_indices["noise"]]
        noise_positions = noise_positions[~mask_indices["noise"]]
        mask_indices["support"] = mask_indices["support"][~mask_indices["noise"]]
        mask_indices["query"] = mask_indices["query"][~mask_indices["noise"]]

        return data, labels, mask_indices, noise_positions

    def loss_acc(self, batch, outlier_batch=None, support_label_noise_choices=[0]):
        ways = self.args.train_way if self.model.training else self.args.test_way
        shot = self.args.train_shot if self.model.training else self.args.test_shot
        query_num = (
            self.args.train_query if self.model.training else self.args.test_query
        )

        # Format data, place on device, add noise
        data, labels, mask_indices, noise_positions = self.prepare_data(
            batch, ways, shot, query_num, support_label_noise_choices, outlier_batch
        )

        # Compute support/query embeddings and prototypes
        outputs = self.model(data, mask_indices["support"])
        support_embeddings = outputs["conv_embeddings"][mask_indices["support"]]
        query_embeddings = outputs["conv_embeddings"][mask_indices["query"]]

        # Note: support labels assumes class labels 0..(N-1), in order. labels has original labels before noise
        support_labels = torch.repeat_interleave(torch.arange(ways), shot).to(
            self.device
        )
        query_labels = labels[mask_indices["query"]].long()

        losses = {}
        accs = {}

        # Prototype-based comparison
        if self.args.comp_method == "proto":
            support_prototypes = outputs["support_prototypes"]

            # Compare query embeddings with prototypes
            logits = (
                self.metric(query_embeddings, support_prototypes)
                / self.args.logit_temperature
            )

            accs["accuracy"] = accuracy(logits, query_labels)
            losses["cross_entropy"] = F.cross_entropy(logits, query_labels)

        # Match query embedding with support embeddings
        elif self.args.comp_method == "match":
            # Create one-hot labels for support set
            one_hot_support_labels = torch.zeros(len(support_labels), ways).to(
                self.device
            )
            one_hot_support_labels[np.arange(len(support_labels)), support_labels] = 1

            # Normalize embeddings and compute cosine distance, then softmax
            support_embeddings_unit = support_embeddings / torch.norm(
                support_embeddings, dim=1, keepdim=True
            )
            query_embeddings_unit = query_embeddings / torch.norm(
                query_embeddings, dim=1, keepdim=True
            )
            similarities = F.softmax(
                query_embeddings_unit @ support_embeddings_unit.T, dim=1
            )

            # Output probability is the similarity scores multiplied with the labels
            p_y = similarities @ one_hot_support_labels

            accs["accuracy"] = accuracy(p_y, query_labels)
            losses["cross_entropy"] = F.nll_loss(p_y.log(), query_labels)

        # Train linear classifier
        elif self.args.comp_method == "lin_cls":
            # Instantiate linear classifier
            lin_cls = nn.Linear(self.model.convnet.conv_out_dims, ways).to(self.device)
            opt = torch.optim.AdamW(lin_cls.parameters(), lr=1e-3, weight_decay=0.01)

            for i in range(101):
                opt.zero_grad()
                lin_y = lin_cls(support_embeddings)
                loss = F.cross_entropy(lin_y, support_labels)
                loss.backward()
                opt.step()

            lin_y_val = lin_cls(query_embeddings)
            accs["accuracy"] = accuracy(lin_y_val, query_labels)

        else:
            raise NotImplementedError

        # Auxiliary loss: protonet of conv embeddings
        if self.args.embed_proto_loss_weight > 0:
            embed_protos = gen_prototypes(support_embeddings, ways, shot)
            embed_protos_logits = metric(query_embeddings, embed_protos)
            embed_protos_loss = self.args.embed_proto_loss_weight * F.cross_entropy(
                embed_protos_logits, query_labels
            )
            losses["conv_embed_proto"] = embed_protos_loss

        # Binary Outlier loss: identify transformer embeddings as noisy samples or not
        if self.args.binary_outlier_loss_weight > 0:
            noise_labels = (
                torch.tensor(noise_positions[mask_indices["support"]])
                .unsqueeze(dim=1)
                .to(self.device)
            )
            accs["binary_outlier"] = accuracy(
                outputs["outlier_logits"], noise_labels, binary=True
            )
            losses["binary_outlier"] = (
                self.args.binary_outlier_loss_weight
                * F.binary_cross_entropy_with_logits(
                    outputs["outlier_logits"], noise_labels
                )
            )

        # Clean Prototype matching: loss to match predicted prototype with clean exmples (reject noisy samples)
        if self.args.clean_proto_loss_weight > 0:
            clean_support_embeddings = support_embeddings[
                ~noise_positions[mask_indices["support"]].astype(bool)
            ]
            num_clean_shots = shot - int(
                noise_positions[mask_indices["support"]].sum() / ways
            )
            clean_protos = gen_prototypes(
                clean_support_embeddings, ways, num_clean_shots
            )

            # Distance of clean protos from predicted protos
            if self.args.dist_metric == "euclidean":
                losses["dist_to_clean_protos"] = (
                    self.args.clean_proto_loss_weight
                    * ((outputs["support_prototypes"] - clean_protos) ** 2).sum()
                    / ways
                )
            elif self.args.dist_metric == "cosine":
                epsilon = 1e-6
                norm_protos = outputs["support_prototypes"] / (
                    torch.norm(outputs["support_prototypes"], dim=1, keepdim=True)
                    + epsilon
                )
                norm_clean_protos = clean_protos / (
                    torch.norm(clean_protos, dim=1, keepdim=True) + epsilon
                )

                # Negative, because we want to maximize cosine angle
                losses["dist_to_clean_protos"] = (
                    -self.args.clean_proto_loss_weight
                    * (norm_protos.flatten() * norm_clean_protos.flatten()).sum()
                    / ways
                )
            else:
                raise NotImplementedError

        return losses, accs

    def nearest_neighbor(
        self, batch, outlier_batch=None, support_label_noise_choices=[0]
    ):
        ways = self.args.train_way if self.model.training else self.args.test_way
        shot = self.args.train_shot if self.model.training else self.args.test_shot
        query_num = (
            self.args.train_query if self.model.training else self.args.test_query
        )

        assert self.args.comp_method.startswith("nearest")
        k = int(self.args.comp_method[8:])  # comp_method specified as "nearest_[k]"

        # Format data, place on device, add noise
        data, labels, mask_indices, noise_positions = self.prepare_data(
            batch, ways, shot, query_num, support_label_noise_choices, outlier_batch
        )

        # Compute support/query embeddings
        outputs = self.model(data, mask_indices["support"])
        support_embeddings = outputs["conv_embeddings"][mask_indices["support"]]
        query_embeddings = outputs["conv_embeddings"][mask_indices["query"]]

        # Note: support labels assumes class labels 0..(N-1), in order
        support_labels = torch.repeat_interleave(torch.arange(ways), shot).to(
            self.device
        )
        query_labels = labels[mask_indices["query"]].long()

        # Find distance between each query and all support samples
        distances = self.metric(query_embeddings, support_embeddings)

        # Sort queries by distance and find labels of closest samples
        sorted_idx = torch.argsort(distances, axis=1, descending=True)
        topk = support_labels[sorted_idx][:, :k]

        # Convert to one-hot
        one_hot_labels = torch.zeros((ways * query_num * k, ways)).to(self.device)
        one_hot_labels[np.arange(ways * query_num * k), topk.flatten()] = 1
        one_hot_labels = one_hot_labels.reshape(ways * query_num, k, ways)

        # Tally votes, add small noise as random tiebreaker
        tallied = (
            one_hot_labels.mean(dim=1)
            + torch.randn((ways * query_num, ways)).to(self.device) / 1e4
        )
        predictions = torch.argmax(tallied, dim=1)

        acc = (predictions == query_labels).sum().float() / query_labels.size(0)

        return {}, {"accuracy": acc}
