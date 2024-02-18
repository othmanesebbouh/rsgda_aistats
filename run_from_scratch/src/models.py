# -*- coding: utf-8 -*-

import os, pprint, tqdm
import numpy as np
import pandas as pd
from haven import haven_utils as hu
from haven import haven_img as hi
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from . import base_classifiers
from . import optimizers

import jax


def get_model_robust(train_loader, exp_dict, device, adv_input_init):
    return RobustClassifier(train_loader, exp_dict, device, adv_input_init)


class RobustClassifier(torch.nn.Module):
    def __init__(self, train_loader, exp_dict, device, adv_input_init):
        super().__init__()

        if (exp_dict["p"] is not None) and (exp_dict["loop_size"] is not None):
            raise ValueError("one of 'p' or 'loop_size' must be None.")

        self.len_train = len(train_loader.dataset)
        self.exp_dict = exp_dict
        self.device = device

        self.model_base = base_classifiers.get_classifier(exp_dict['model'])
        self.n_params = sum(p.numel() for p in self.model_base.parameters() if p.requires_grad)
        self.model_name = exp_dict['model']

        self.adv_input = adv_input_init.to(device)
        self.to(device=self.device)

        # Load optimizers
        self.opt = torch.optim.SGD(self.parameters(),
                                   lr=exp_dict['lr_d'])
        self.gamma = exp_dict["gamma"]
        self.n_it = 0
        self.epoch = 1
        self.grad_norm = 0
        self.min_grad_norm = 0
        self.lr_d = exp_dict["lr_d"]
        self.lr_a = exp_dict["lr_a"]
        self.lr_compute_phi = exp_dict["lr_compute_phi"]
        self.batch_size = exp_dict['batch_size']
        self.it_per_epoch = self.len_train // self.batch_size + 1
        self.p = exp_dict["p"]
        self.loop_size = exp_dict["loop_size"]
        self.coin_flips = None

    def train_on_loader(self, train_loader, epoch):
        self.train()
        pbar = tqdm.tqdm(train_loader)
        for batch_number, batch in enumerate(pbar):
            score_dict = self.train_on_batch(batch, batch_number, epoch)
            pbar.set_description('Train loss - {:.3f}, Train grad norm -  {:.3f}'.format(score_dict["train_loss"],
                                                                                         score_dict["train_grad"]))
        return score_dict

    def get_state_dict(self):
        state_dict = {"model": self.model_base.state_dict(),
                      "opt": self.opt.state_dict()}

        return state_dict

    def set_state_dict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"])
        self.opt.load_state_dict(state_dict["opt"])

    def val_on_dataset(self, dataset, metric, name, train=False):
        self.eval()

        metric_function = get_metric_function(metric)
        loader = torch.utils.data.DataLoader(dataset, drop_last=False,
                                             batch_size=self.exp_dict['batch_size'])

        score_sum = 0.
        pbar = tqdm.tqdm(loader)

        self.unfreeze_weights()
        grad_norm = 0.
        adv_train_loss = 0.
        adv_quad_loss = 0.
        adv_grad = 0.

        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        adv_criterion = torch.nn.MSELoss(reduction='mean')

        for batch_number, batch in enumerate(pbar):
            images = batch["images"].to(device=self.device)
            labels = batch["labels"].to(device=self.device)
            score_sum += metric_function(self.model_base, images,
                                         labels).item() * images.shape[0]
            score = float(score_sum / len(loader.dataset))
            pbar.set_description(f'Validating {metric}: {score:.3f}')

            if train:
                indices = batch["meta"]["indices"].to(device=self.device)
                adv_images = self.adv_input[indices].clone().detach().requires_grad_(True)

                loss = criterion(self.model_base.forward(adv_images), labels.long().view(-1)) * images.shape[0]
                quad_loss = adv_criterion(images, adv_images) * images.shape[-1] ** 2 * images.shape[0]
                adv_train_loss += loss / len(loader.dataset)
                adv_quad_loss += quad_loss / len(loader.dataset)

                total_loss = loss - self.gamma * quad_loss
                total_loss.backward()

                # grad_y
                adv_grad_norm = adv_images.grad.data.norm(2)
                adv_grad += adv_grad_norm.item() ** 2

        if train:
            grad_norm = 0
            for p in self.model_base.parameters():
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2

            grad_norm = grad_norm / (len(loader.dataset) ** 2)
            adv_grad = adv_grad / (len(loader.dataset) ** 2)

            total_grad = grad_norm + adv_grad

            if self.epoch == 1:
                self.min_grad_norm = total_grad
            else:
                self.min_grad_norm = min(self.min_grad_norm, total_grad)
            self.epoch += 1

        if train:
            return {f'{dataset.split}_{name}': score, 'train_grad': float(grad_norm),
                    'adv_grad': float(adv_grad), 'adv_train_loss': float(adv_train_loss),
                    'quad_loss': float(adv_quad_loss)}
        else:
            return {f'{dataset.split}_{name}': score}

    def grad_phi(self, dataset, check_every, max_epoch_phi, tol=1e-10):
        self.freeze_weights()
        adv_grad = tol + 1
        adv_input_monitoring = self.adv_input.clone().detach().requires_grad_(False)
        step = 0
        ct = 0
        while adv_grad > tol and step < max_epoch_phi:
            step += 1
            # do gradient descent ascent for 'check_every' epochs
            for e in range(check_every):
                adv_input_monitoring = self.compute_total_loss(dataset, comp_grad_phi=False,
                                                               adv_input_monitoring=adv_input_monitoring,
                                                               optimize=True, comp_adv_grad_norm=False,
                                                               comp_adv_losses=False, step=(e + 1) * step)[
                    0].clone().detach().requires_grad_(False)
                ct += 1

            # compute the adversarial gradient to check for convergence
            adv_grad = self.compute_total_loss(dataset, comp_grad_phi=False, adv_input_monitoring=adv_input_monitoring,
                                               optimize=False, comp_adv_grad_norm=True,
                                               comp_adv_losses=False, step=step)[0]

            print('Number of epochs to compute grad(phi)(x) so far = ', ct)
            print('norm(grad_y(f(x,y))) = ', adv_grad)

        print('Total number of epochs to compute grad(phi(x)) = ', check_every * step)

        self.unfreeze_weights()
        # once we have converged, we can compute the gradient with respect to the parameters
        # and the true loss
        grad_phi_norm, phi = \
            self.compute_total_loss(dataset, comp_grad_phi=True, adv_input_monitoring=adv_input_monitoring,
                                    optimize=False, comp_adv_grad_norm=False,
                                    comp_adv_losses=False, step=step)[0]

        if step == max_epoch_phi:
            grad_phi_norm = np.inf

        return {'grad_phi': float(grad_phi_norm), 'phi': float(phi)}

    def compute_total_loss(self, dataset, adv_input_monitoring, comp_grad_phi=False, optimize=False,
                           comp_adv_grad_norm=False, comp_adv_losses=False, step=0):
        loader = torch.utils.data.DataLoader(dataset, drop_last=False,
                                             batch_size=self.exp_dict['batch_size'])

        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        adv_criterion = torch.nn.MSELoss(reduction='mean')
        if comp_grad_phi:
            phi = 0

        if comp_adv_grad_norm:
            adv_grad = 0
        if comp_adv_losses:
            adv_train_loss = 0
            adv_quad_loss = 0

        result = []
        pbar = tqdm.tqdm(loader)
        if optimize:
            pbar.set_description("Doing gradient ascent, epoch {}".format(step))
        if comp_adv_grad_norm:
            pbar.set_description("Computing adversarial gradient")

        for batch_number, batch in enumerate(pbar):
            images = batch["images"].to(device=self.device)
            labels = batch["labels"].to(device=self.device)
            indices = batch["meta"]["indices"].to(device=self.device)
            adv_images = adv_input_monitoring[indices].clone().detach().requires_grad_(True)

            loss = criterion(self.model_base.forward(adv_images), labels.long().view(-1))
            quad_loss = adv_criterion(images, adv_images) * images.shape[-1] ** 2
            total_loss = loss - self.gamma * quad_loss
            if not optimize:
                total_loss *= images.shape[0]

            total_loss.backward()

            if comp_adv_losses:
                adv_train_loss += loss / len(loader.dataset)
                adv_quad_loss += quad_loss / len(loader.dataset)

            if optimize:
                adv_images.data.add_(self.lr_compute_phi * adv_images.grad.data)
                adv_input_monitoring[indices] = adv_images.clone().detach().requires_grad_(False)

            if comp_adv_grad_norm:
                adv_grad_norm = adv_images.grad.data.norm(2)
                adv_grad += adv_grad_norm.item() ** 2 / len(loader.dataset) ** 2

            if comp_grad_phi:
                phi += total_loss / len(loader.dataset)

        if comp_grad_phi:
            grad_norm = 0
            for p in self.model_base.parameters():
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2 / len(loader.dataset) ** 2
            result.append((grad_norm, phi))

        if optimize:
            result.append(adv_input_monitoring)

        if comp_adv_grad_norm:
            result.append(adv_grad)

        if comp_adv_losses:
            result.append((adv_train_loss, adv_quad_loss))

        return result

    def val_on_loader(self, batch):
        pass

    def train_on_batch(self, batch, batch_number, epoch):
        images = batch["images"].to(device=self.device)
        labels = batch["labels"].to(device=self.device)
        indices = batch["meta"]["indices"].to(device=self.device)
        if self.exp_dict['dataset'] == 'mnist2d':
            if epoch == 0 and batch_number == 0:
                self.adv_input = self.adv_input.reshape([len(self.adv_input)]
                                                        + list(images.shape[1:]))
            if epoch == 0:
                self.adv_input[indices] = images.clone().detach()
        if batch_number == 0 and self.loop_size is None:
            self.coin_flips = np.random.binomial(n=1, p=self.p, size=self.it_per_epoch)

        self.n_it += 1
        if self.p is None:
            if self.loop_size > 0:
                if self.n_it % self.loop_size == 0:
                    loss = self.descent_step(indices, labels)
                else:
                    loss = self.ascent_step(indices, images, labels)
            else:
                if self.n_it % (-self.loop_size) == 0:
                    loss = self.ascent_step(indices, images, labels)
                else:
                    loss = self.descent_step(indices, labels)
        else:
            if self.coin_flips[batch_number]:
                loss = self.descent_step(indices, labels)
            else:
                loss = self.ascent_step(indices, images, labels)

        self.grad_norm = 0
        for p in self.model_base.parameters():
            param_norm = p.grad.data.norm(2)
            self.grad_norm += param_norm.item() ** 2

        return {'train_loss': float(loss), 'train_grad': float(self.grad_norm)}

    def descent_step(self, indices, labels):
        self.opt.zero_grad()
        self.unfreeze_weights()

        adv_images = self.adv_input[indices].clone().detach().requires_grad_(False)
        closure = lambda: softmax_loss(self.model_base, adv_images, labels,
                                       backwards=True, reduction="mean")
        loss = self.opt.step(closure=closure)
        self.freeze_weights()
        return loss

    def ascent_step(self, indices, images, labels):
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        adv_images = self.adv_input[indices].clone().detach().requires_grad_(True)
        adv_criterion = torch.nn.MSELoss(reduction="mean")
        loss = criterion(self.model_base.forward(adv_images), labels.long().view(-1))
        quad_loss = adv_criterion(images, adv_images) * images.shape[-1] ** 2
        total_loss = loss - self.gamma * quad_loss

        # backward
        total_loss.backward()
        adv_images.data.add_(self.lr_a * adv_images.grad.data)

        # copy adv_images data into self.adv_input
        self.adv_input[indices] = adv_images.clone().detach().requires_grad_(False)
        return total_loss

    def freeze_weights(self):
        if self.model_name == "pre_trained_resnet50":
            for param in list(self.parameters())[-4:]:
                param.requires_grad = True
        else:
            for param in self.parameters():
                param.requires_grad = False

    def unfreeze_weights(self):
        if self.model_name == "pre_trained_resnet50":
            for param in list(self.parameters())[-4:]:
                param.requires_grad = True
        else:
            for param in self.parameters():
                param.requires_grad = True


# Metrics
def get_metric_function(metric):
    if metric == "softmax_accuracy":
        return softmax_accuracy

    elif metric == "softmax_loss":
        return softmax_loss


def softmax_loss(model, images, labels, backwards=False, reduction="mean"):
    logits = model(images)
    criterion = torch.nn.CrossEntropyLoss(reduction=reduction)
    loss = criterion(logits, labels.long().view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss


def adv_loss(model, images, labels, indices, gamma=0.5,
             backwards=False, reduction="mean"):
    adv_images = model.adv_input[indices]
    adv_logits = model.model_base(adv_images)

    criterion = torch.nn.CrossEntropyLoss(reduction=reduction)
    adv_criterion = torch.nn.MSELoss(reduction=reduction)
    objective = criterion(adv_logits, labels.long().view(-1))
    quad_loss = adv_criterion(images, adv_images)
    loss = gamma * quad_loss - objective

    if backwards and loss.requires_grad:
        loss.backward()

    return loss


def softmax_accuracy(model, images, labels):
    logits = model(images)
    pred_labels = logits.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()

    return acc
