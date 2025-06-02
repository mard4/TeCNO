import logging
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
#from pytorch_lightning.metrics.utils import _input_format_classification
from pytorch_lightning.core.lightning import LightningModule
from utils.metric_helper import AccuracyStages, RecallOverClasse, PrecisionOverClasses
from torch import nn
import numpy as np


class TeCNO(LightningModule):
    def __init__(self, hparams, model, dataset):
        super(TeCNO, self).__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.dataset = dataset
        self.model = model
        self.weights_train = np.asarray(self.dataset.weights["train"])
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.from_numpy(self.weights_train).float())
        self.init_metrics()

    def init_metrics(self):
        self.train_acc_stages = AccuracyStages(num_stages=self.hparams.mstcn_stages)
        self.val_acc_stages = AccuracyStages(num_stages=self.hparams.mstcn_stages)
        self.max_acc_last_stage = {"epoch": 0, "acc": 0}
        self.max_acc_global = {"epoch": 0, "acc": 0 , "stage": 0, "last_stage_max_acc_is_global": False}

        self.precision_metric = PrecisionOverClasses(num_classes=7)
        self.recall_metric = RecallOverClasse(num_classes=7)
        #self.cm_metric = ConfusionMatrix(num_classes=10)

    def forward(self, x):
        video_fe = x.transpose(2, 1)
        y_classes = self.model.forward(video_fe)
        y_classes = torch.softmax(y_classes, dim=2)
        return y_classes

    def loss_function(self, y_classes, labels):
        stages = y_classes.shape[0]
        clc_loss = 0
        for j in range(stages):  ### make the interuption free stronge the more layers.
            p_classes = y_classes[j].squeeze().transpose(1, 0)
            ce_loss = self.ce_loss(p_classes, labels.squeeze())
            clc_loss += ce_loss
        clc_loss = clc_loss / (stages * 1.0)
        return clc_loss

    def get_class_acc(self, y_true, y_classes):
        y_true = y_true.squeeze()
        y_classes = y_classes.squeeze()
        y_classes = torch.argmax(y_classes, dim=0)
        acc_classes = torch.sum(
            y_true == y_classes).float() / (y_true.shape[0] * 1.0)
        return acc_classes

    def get_class_acc_each_layer(self, y_true, y_classes):
        y_true = y_true.squeeze()
        accs_classes = []
        for i in range(y_classes.shape[0]):
            acc_classes = self.get_class_acc(y_true, y_classes[i, 0])
            accs_classes.append(acc_classes)
        return accs_classes


    '''def log_precision_and_recall(self, precision, recall, step):
        for n,p in enumerate(precision):
            if not p.isnan():
                self.log(f"{step}_precision_{self.dataset.class_labels[n]}",p ,on_step=True, on_epoch=True)
        for n,p in enumerate(recall):
            if not p.isnan():
                self.log(f"{step}_recall_{self.dataset.class_labels[n]}",p ,on_step=True, on_epoch=True)'''

    def calc_precision_and_recall(self, y_pred, y_true, step="val"):
        """
        Instead of using _input_format_classification (which doesn’t exist in PL 0.8.5),
        we simply take argmax over the class‐dimension for each frame.  y_pred[-1] should
        have shape [batch_size, num_classes, T], and y_true should be [batch_size, T].
        """
        # y_pred[-1] → shape: [batch, num_classes, T]
        # We want per‐frame class indices: argmax over dim=1 → [batch, T]
        y_max_pred = torch.argmax(y_pred[-1], dim=1)

        # y_true is already [batch, T] of integer class labels, so no change:
        # (If y_true has a shape like [batch, 1, T], you may need to squeeze: y_true = y_true.squeeze(1))
        if y_true.dim() == 3 and y_true.size(1) == 1:
            # sometimes y_true is [batch, 1, T].  Make it [batch, T]:
            y_true = y_true.squeeze(1)

        # Now compute precision/recall with torchmetrics:
        precision = self.precision_metric(y_max_pred, y_true)
        recall    = self.recall_metric(y_max_pred, y_true)
        return precision, recall



    def log_average_precision_recall(self, outputs, step="val"):
        """
        Computes average precision and recall over all batches for the given step.
        Each element `o` in `outputs` must contain:
          - f"{step}_precision"  (Tensor of shape [num_classes] or a scalar)
          - f"{step}_recall"     (Tensor of shape [num_classes] or a scalar)
        We then average across batches and print/log them.
        """

        # 1) Construct the keys depending on the step:
        prec_key = f"{step}_precision"
        rec_key  = f"{step}_recall"

        # 2) Collect only those outputs that contain the keys:
        precision_list = [o[prec_key] for o in outputs if prec_key in o]
        recall_list    = [o[rec_key]  for o in outputs if rec_key  in o]

        if len(precision_list) == 0 or len(recall_list) == 0:
            # No precision/recall to aggregate
            return

        # 3) Stack them into a single tensor:
        x = torch.stack(precision_list)  # shape could be (B, C) or (B,) or (B, 1)
        y = torch.stack(recall_list)

        # 4) If x is 1D (single‐class or scalar), just take a simple mean:
        if x.ndim == 1:
            # Each o[prec_key] was a scalar => x is shape (num_batches,)
            avg_precision_over_video = x.mean()
        else:
            # Multi‐class case: x has shape (B, num_classes)
            num_classes = x.shape[1]
            class_wise_avgs = []
            for n in range(num_classes):
                # Filter out any NaNs for this class dimension:
                col = x[:, n]
                valid = col[~col.isnan()]
                if valid.numel() > 0:
                    class_wise_avgs.append(valid.mean())
                else:
                    class_wise_avgs.append(torch.tensor(float("nan")))

            class_wise_avgs = torch.stack(class_wise_avgs)  # shape: (num_classes,)
            valid_aw = class_wise_avgs[~class_wise_avgs.isnan()]
            if valid_aw.numel() > 0:
                avg_precision_over_video = valid_aw.mean()
            else:
                avg_precision_over_video = torch.tensor(float("nan"))

        # 5) Do the same for recall:
        if y.ndim == 1:
            avg_recall_over_video = y.mean()
        else:
            num_classes = y.shape[1]
            class_wise_avgs = []
            for n in range(num_classes):
                col = y[:, n]
                valid = col[~col.isnan()]
                if valid.numel() > 0:
                    class_wise_avgs.append(valid.mean())
                else:
                    class_wise_avgs.append(torch.tensor(float("nan")))

            class_wise_avgs = torch.stack(class_wise_avgs)
            valid_ar = class_wise_avgs[~class_wise_avgs.isnan()]
            if valid_ar.numel() > 0:
                avg_recall_over_video = valid_ar.mean()
            else:
                avg_recall_over_video = torch.tensor(float("nan"))

        # 6) Finally, print the results. Lightning 0.8.5 does not use self.log here:
        print(
            f"[{step.capitalize()} Epoch] "
            f"avg_precision: {avg_precision_over_video:.4f}, "
            f"avg_recall:    {avg_recall_over_video:.4f}"
        )



    def training_step(self, batch, batch_idx):
        """
        Called for each training batch. We must return a dict containing:
          - "loss": the loss Tensor (Lightning uses this to call backward())
          - any other scalars you want Lightning to log (e.g. precision, recall, stage‐accuracies)

        Here we:
          1) Do a forward pass
          2) Compute cross‐entropy loss
          3) Compute precision, recall, and per‐stage accuracies
          4) Return a dict of all these values
        """
        stem, y_hat, y_true = batch
        y_pred = self.forward(stem)

        # 1) Compute classification loss over all stages:
        loss = self.loss_function(y_pred, y_true)

        # 2) Compute precision & recall (micro) over last‐stage logits vs. true labels:
        precision, recall = self.calc_precision_and_recall(y_pred, y_true, step="train")

        # 3) Compute per‐stage accuracies and rename final stage to "train_acc":
        acc_list = self.train_acc_stages(y_pred, y_true)
        # Build a dict: { "train_S1_acc": acc_list[0], "train_S2_acc": acc_list[1], ... }
        stage_metrics = { f"train_S{s+1}_acc": acc_list[s] for s in range(len(acc_list)) }
        # Rename last stage’s entry to "train_acc"
        stage_metrics["train_acc"] = stage_metrics.pop(f"train_S{len(acc_list)}_acc")

        # 4) Return a dict containing "loss" plus all metrics. Lightning 0.8.5
        #    will log any scalar keys automatically.
        return {
            "loss": loss,
            "train_precision": precision,
            "train_recall": recall,
            **stage_metrics
        }


    def training_epoch_end(self, outputs):
        """
        Called at the end of the training epoch. We compute/print average precision & recall,
        then return an empty dict so Lightning doesn’t crash.
        """
        self.log_average_precision_recall(outputs, step="train")

        # Return a dict (even if empty) so Lightning does not see `None`.
        return {}



    def validation_step(self, batch, batch_idx):
        """
        Called for each validation batch. Return a dict with:
          - "val_loss"       (Tensor)
          - "val_precision"  (Tensor)
          - "val_recall"     (Tensor)
          - "val_S1_acc", ..., and final "val_acc"  (floats)

        Lightning will collect these across all val batches.
        """
        stem, y_hat, y_true = batch
        y_pred = self.forward(stem)

        # 1) Compute validation loss
        val_loss = self.loss_function(y_pred, y_true)

        # 2) Compute precision & recall (Tensors) over last-stage logits
        precision, recall = self.calc_precision_and_recall(y_pred, y_true, step="val")

        # 3) Compute per-stage accuracies (floats)
        acc_list = self.val_acc_stages(y_pred, y_true)
        stage_metrics = {f"val_S{s+1}_acc": acc_list[s] for s in range(len(acc_list))}
        # Rename the last stage’s accuracy to "val_acc"
        stage_metrics["val_acc"] = stage_metrics.pop(f"val_S{len(acc_list)}_acc")

        # 4) Return all scalars in a dict
        return {
            "val_loss": val_loss,
            "val_precision": precision,
            "val_recall": recall,
            **stage_metrics
        }


    def validation_epoch_end(self, outputs):
        """
        Called once at the end of validation. `outputs` is a list of dicts from validation_step,
        each containing keys:
          - "val_loss"       (Tensor)
          - "val_precision"  (Tensor)
          - "val_recall"     (Tensor)
          - "val_S1_acc", ..., "val_acc"  (floats)

        We need to:
          a) Average all "val_acc" floats to get val_acc_stage_last_epoch.
          b) Update self.max_acc_last_stage if this epoch’s average is higher.
          c) Print/log self.max_acc_last_stage.
          d) Return {"val_acc": val_acc_stage_last_epoch} so that Lightning’s EarlyStopping sees it.
          e) (Optionally) aggregate and print avg precision/recall.
        """

        # (a) Average over all "val_acc" floats from each batch
        val_acc_values = [o["val_acc"] for o in outputs if "val_acc" in o]
        if len(val_acc_values) > 0:
            val_acc_stage_last_epoch = sum(val_acc_values) / len(val_acc_values)
        else:
            val_acc_stage_last_epoch = 0.0

        # (b) Update max if needed
        if val_acc_stage_last_epoch > self.max_acc_last_stage["acc"]:
            self.max_acc_last_stage["acc"] = val_acc_stage_last_epoch
            self.max_acc_last_stage["epoch"] = self.current_epoch

        # (c) Print the current maximum
        print(f"[Epoch {self.current_epoch}] max_acc_last_stage: {self.max_acc_last_stage['acc']:.4f}")

        # (d) Aggregate average precision/recall (Tensors) and print
        self.log_average_precision_recall(outputs, step="val")

        # (e) **Return** the val_acc so that Lightning logs it and EarlyStopping can use it
        return {
            "val_acc": torch.tensor(val_acc_stage_last_epoch)
        }



    def test_step(self, batch, batch_idx):
        """
        Called for each test batch. Return a dict with:
          - "test_loss"
          - "test_precision", "test_recall"
          - "test_S1_acc", "test_S2_acc", ..., and final "test_acc"

        Lightning will aggregate these across all test batches.
        """
        stem, y_hat, y_true = batch
        y_pred = self.forward(stem)

        # 1) Compute test loss:
        test_loss = self.loss_function(y_pred, y_true)

        # 2) Compute precision & recall over last‐stage logits vs. true labels:
        precision, recall = self.calc_precision_and_recall(y_pred, y_true, step="test")

        # 3) Compute per‐stage accuracies and rename last stage to "test_acc":
        acc_list = self.val_acc_stages(y_pred, y_true)  # using val_acc_stages for computing stage‐accuracies
        stage_metrics = { f"test_S{s+1}_acc": acc_list[s] for s in range(len(acc_list)) }
        stage_metrics["test_acc"] = stage_metrics.pop(f"test_S{len(acc_list)}_acc")

        # 4) Return everything in a dict:
        return {
            "test_loss": test_loss,
            "test_precision": precision,
            "test_recall": recall,
            **stage_metrics
        }


    def test_epoch_end(self, outputs):
        """
        Called once after all test_step calls. `outputs` is a list of dicts returned
        by test_step. Each dict should contain:
          - "test_S1_acc", "test_S2_acc", ..., and final "test_acc"  (floats)
          - "test_precision", "test_recall"  (Tensors)
        We need to:
          a) Average all "test_acc" floats to get test_acc_overall.
          b) Print or log that test accuracy.
          c) Compute and print average precision/recall if desired.
          d) Return {"test_acc": Tensor(...)} so that Lightning can log it.
        """

        # (a) Average over all test_acc floats from each batch
        test_acc_values = [o["test_acc"] for o in outputs if "test_acc" in o]
        if len(test_acc_values) > 0:
            test_acc_overall = sum(test_acc_values) / len(test_acc_values)
        else:
            test_acc_overall = 0.0

        # (b) Print the overall test accuracy
        print(f"[Test] test_acc: {test_acc_overall:.4f}")

        # (c) If your outputs contain "test_precision" / "test_recall" (Tensors),
        #     you can call the same helper to aggregate them:
        self.log_average_precision_recall(outputs, step="test")

        # (d) Return the test_acc in a Tensor so Lightning logs it properly:
        return {
            "test_acc": torch.tensor(test_acc_overall)
        }




    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate)
        return [optimizer]  #, [scheduler]

    def __dataloader(self, split=None):
        dataset = self.dataset.data[split]
        should_shuffle = False
        if split == "train":
            should_shuffle = True
        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = None
        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)
            should_shuffle = False
        print(f"split: {split} - shuffle: {should_shuffle}")
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        return loader

    def train_dataloader(self):
        dataloader = self.__dataloader(split="train")
        logging.info("training data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    def val_dataloader(self):
        dataloader = self.__dataloader(split="val")
        logging.info("validation data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    def test_dataloader(self):
        dataloader = self.__dataloader(split="test")
        logging.info("test data loader called  - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    @staticmethod
    def add_module_specific_args(parser):  # pragma: no cover
        regressiontcn = parser.add_argument_group(
            title='regression tcn specific args options')
        regressiontcn.add_argument("--learning_rate",
                                   default=0.001,
                                   type=float)
        regressiontcn.add_argument("--optimizer_name",
                                   default="adam",
                                   type=str)
        regressiontcn.add_argument("--batch_size", default=1, type=int)

        return parser
