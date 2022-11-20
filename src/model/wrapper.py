import os
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from model import MyModel
import matplotlib.pyplot as plt
import util


class Wrapper(pl.LightningModule):
    def __init__(self, config: dict, minibatch_size: int):
        """A lightning wrapper for a CLIP model as specified in the paper.
        Args:
            config (dict): A dictionary containing the CLIP instantiation parameters.
        """
        super().__init__()

        self.model = MyModel()
        self.minibatch_size = minibatch_size
        # self.automatic_optimization = False # 手动优化
        self.train_dataset_size = config["train_data_size"]
        self.config = config
        self.ts = time.time()
        self.counter = 0
        self.myutil = util.UtilClass()

    # Sourced from https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset_size = self.train_dataset_size
        num_devices = max(1, self.trainer.num_devices)
        effective_batch_size = self.minibatch_size * self.trainer.accumulate_grad_batches * num_devices
        return (dataset_size // effective_batch_size) * self.trainer.max_epochs

    def get_loss(self, out, trg, criteria, mask, reduction='mean', type="MSE"):
        pos_weight = mask.view(-1, 224 * 224)
        if (type == "MSE"):
            loss = criteria(out, trg, pos_weight)
        elif (type == "BCE"):
            loss = criteria(out, trg, pos_weight=pos_weight, reduction=reduction)
        else:
            loss = criteria(out, trg, reduction=reduction)
        return loss

    # https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html
    def training_step(self, train_batch, batch_idx):
        input_image, target_image, mask = train_batch
        output_image = self.model(input_image)
        if (self.config["loss_type"] == "BCE"):
            criteria = F.binary_cross_entropy_with_logits
        elif (self.config["loss_type"] == "L1"):
            criteria = F.l1_loss
        elif (self.config["loss_type"] == "MSE"):
            criteria = self.weighted_mse_loss

        loss = self.get_loss(output_image.view(-1, 224 * 224), target_image.view(-1, 224 * 224),
                             criteria=criteria, mask=mask, reduction='mean', type=self.config["loss_type"])
        cmf, _, _, _ = self.myutil.get_acc(config=self.config,
                                           src_tensor=input_image[:, 1],
                                           output_tensor=output_image[:, 0],
                                           batch_idx=batch_idx,
                                           batch_size=self.minibatch_size,
                                           data_type="train",
                                           cmf_flag=True,
                                           other_flag=False)
        self.log_dict({
            'train_loss': loss,
            'train_cmf': cmf
        }, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        input_image, target_image, mask = val_batch
        output_image = self.model(input_image)

        if (self.config["loss_type"] == "BCE"):
            criteria = F.binary_cross_entropy_with_logits
        elif (self.config["loss_type"] == "L1"):
            criteria = F.l1_loss
        elif (self.config["loss_type"] == "MSE"):
            criteria = self.weighted_mse_loss

        loss = self.get_loss(output_image.view(-1, 224 * 224), target_image.view(-1, 224 * 224),
                             criteria=criteria, mask=mask,
                             reduction='mean')
        cmf, _, _, _ = self.myutil.get_acc(config=self.config,
                                           src_tensor=input_image[:, 1],
                                           output_tensor=output_image[:, 0],
                                           batch_idx=batch_idx,
                                           batch_size=self.minibatch_size,
                                           data_type="train",
                                           cmf_flag=True,
                                           other_flag=False)
        self.log_dict({
            'val_loss': loss,
            'val_cmf': cmf
        }, prog_bar=True)

        if (batch_idx == 0):
            # save a example picture
            plt.figure(figsize=(4, 4))

            plt.subplot(2, 2, 0 + 1)
            plt.imshow(input_image[0, 0].view(224, 224).cpu().data.numpy())
            plt.axis('off')

            plt.subplot(2, 2, 1 + 1)
            plt.imshow(input_image[0, 1].view(224, 224).cpu().data.numpy())
            plt.axis('off')

            plt.subplot(2, 2, 2 + 1)
            plt.imshow(target_image[0, 0].view(224, 224).cpu().data.numpy())
            plt.axis('off')

            plt.subplot(2, 2, 3 + 1)
            plt.imshow(output_image[0, 0].view(224, 224).cpu().data.numpy())
            plt.axis('off')

            if not os.path.exists(os.path.join("../../data/train_visualized", str(self.ts))):
                if not (os.path.exists(os.path.join("../../data/train_visualized"))):
                    os.mkdir(os.path.join("../../data/train_visualized"))
                os.mkdir(os.path.join("../../data/train_visualized", str(self.ts)))

            plt.savefig(
                os.path.join("../../data/train_visualized", str(self.ts),
                             "I{:d}.png".format(self.counter)),
                dpi=300)
            plt.clf()
            plt.close('all')
            self.counter += 1

        return {"loss": loss}

    def test_step(self, test_batch, batch_idx):
        input_image, target_image, mask = test_batch
        output_image = self.model(input_image)[:, 0]
        src_image = input_image[:, 1]
        return src_image, output_image

    def test_epoch_end(self, outputs) -> None:
        src_lst = [i[0] for i in outputs]
        output_lst = [i[1] for i in outputs]

        src_tensor = torch.cat(src_lst, dim=0).cpu()
        output_tensor = torch.cat(output_lst, dim=0).cpu()

        if not os.path.exists("./data/save_output/"):
            os.mkdir("./data/save_output/")
        torch.save(src_tensor, f"./data/save_output/src_tensor")
        torch.save(output_tensor, f"./data/save_output/output_tensor")

        cmf, rmf, precision, recall = self.myutil.get_acc(config=self.config,
                                                          src_tensor=src_tensor,
                                                          output_tensor=output_tensor,
                                                          batch_idx=0,
                                                          batch_size=
                                                          # 30,
                                                          src_tensor.shape[0],
                                                          data_type="val",
                                                          cmf_flag=True,
                                                          other_flag=True)
        print(f"cmf:{cmf}, rmf:{rmf}, precision:{precision}, recall:{recall}")

    def forward(self, input_image):
        ourput_image = self.model(input_image)
        return ourput_image

    def configure_optimizers(self):
        lr = self.config["learning_rate"]

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(
                0.9,
                0.98
            ),
            eps=1e-6,
            weight_decay=0.2
        )

        # Use pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
        print(f"self.num_training_steps:{self.num_training_steps}")
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.num_training_steps,
            cycle_mult=1.0,
            max_lr=lr,
            min_lr=0,
            warmup_steps=100
        )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def weighted_mse_loss(self, input, target, weight):
        return (weight * (input - target) ** 2).sum() / weight.sum()
