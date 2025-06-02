import logging
import torch
from torch import optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from pycm import ConfusionMatrix
import numpy as np
import pickle
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy,
    MultilabelAccuracy,
    MultilabelF1Score
)



class FeatureExtraction(LightningModule):
    def __init__(self, hparams, model, dataset):
        super(FeatureExtraction, self).__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.dataset = dataset
        self.num_tasks = self.hparams.num_tasks  # output stem 0, output phase 1 , output phase and tool 2
        self.log_vars = nn.Parameter(torch.zeros(2))
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.from_numpy(self.dataset.class_weights).float())
        self.sig_f = nn.Sigmoid()
        self.current_video_idx = self.dataset.df["test"].video_idx.min()
        self.init_metrics()

        # store model
        self.current_stems = []
        self.current_phase_labels = []
        self.current_p_phases = []
        self.len_test_data = len(self.dataset.data["test"])
        self.model = model
        self.best_metrics_high = {"val_acc_phase": 0}
        self.test_acc_per_video = {}
        self.pickle_path = None


            
    def init_metrics(self):
        
        # multiclass phase‐accuracy (7 classes)
        self.train_acc_phase = MulticlassAccuracy(num_classes=self.hparams.out_features)
        self.val_acc_phase   = MulticlassAccuracy(num_classes=self.hparams.out_features)
        self.test_acc_phase  = MulticlassAccuracy(num_classes=self.hparams.out_features)
        if self.num_tasks == 2:
                # multilabel tool‐accuracy & F1 (7 tools)
                self.train_acc_tool = MultilabelAccuracy(num_labels=self.hparams.out_features)
                self.val_acc_tool   = MultilabelAccuracy(num_labels=self.hparams.out_features)
                self.train_f1_tool  = MultilabelF1Score(num_labels=self.hparams.out_features)
                self.val_f1_tool    = MultilabelF1Score(num_labels=self.hparams.out_features)

            
            
            
    def set_export_pickle_path(self):
        self.pickle_path = self.hparams.output_path / "cholec80_pickle_export"
        self.pickle_path.mkdir(exist_ok=True)
        print(f"setting export_pickle_path: {self.pickle_path}")

    # ---------------------
    # TRAINING
    # ---------------------

    def forward(self,x):
        stem, phase, tool = self.model.forward(x)
        return stem, phase, tool

    def loss_phase_tool(self, p_phase, p_tool, labels_phase, labels_tool, num_tasks):
        loss_phase = self.ce_loss(p_phase, labels_phase)
        if num_tasks == 1:
            return loss_phase
        # else
        labels_tool = torch.stack(labels_tool, dim=1)
        loss_tools = self.bce_loss(p_tool, labels_tool.data.float())
        # automatic balancing
        precision1 = torch.exp(-self.log_vars[0])
        loss_phase_l = precision1 * loss_phase + self.log_vars[0]
        precision2 = torch.exp(-self.log_vars[1])
        loss_tool_l = precision2 * loss_tools + self.log_vars[1]
        loss = loss_phase_l + loss_tool_l
        return loss



    def training_step(self, batch, batch_idx):
        x, y_phase, y_tool = batch
        _, p_phase, p_tool = self.forward(x)
        loss = self.loss_phase_tool(p_phase, p_tool, y_phase, y_tool, self.num_tasks)
        # acc_phase, acc_tool, loss
        logs = {}
        if self.num_tasks == 2:
            self.train_acc_tool(p_tool, torch.stack(y_tool, dim=1))
            self.train_f1_tool(p_tool, torch.stack(y_tool, dim=1))
            logs["train_acc_tool"] = self.train_acc_tool.compute()
            logs["train_f1_tool"]  = self.train_f1_tool.compute()
        self.train_acc_phase(p_phase, y_phase)
        logs["train_acc_phase"] = self.train_acc_phase.compute()
        logs["loss"] = loss
 
        # return dict for Lightning 0.8.5
        return {
            "loss": loss,
            "log": logs,
            #"progress_bar": {"loss": loss}
        }




    def validation_step(self, batch, batch_idx):
        x, y_phase, y_tool = batch
        _, p_phase, p_tool = self.forward(x)
        loss = self.loss_phase_tool(p_phase, p_tool, y_phase, y_tool, self.num_tasks)
        # acc_phase, acc_tool, loss
        logs = {}
        if self.num_tasks == 2:
            self.val_acc_tool(p_tool, torch.stack(y_tool, dim=1))
            self.val_f1_tool(p_tool, torch.stack(y_tool, dim=1))
            logs["val_acc_tool"] = self.val_acc_tool.compute()
            logs["val_f1_tool"]  = self.val_f1_tool.compute()
        self.val_acc_phase(p_phase, y_phase)
        logs["val_acc_phase"] = self.val_acc_phase.compute()
        logs["val_loss"]      = loss
 
        return {
            "val_loss": loss,
            "val_log": logs}

    def get_phase_acc(self, true_label, pred):
        pred = torch.FloatTensor(pred)
        pred_phase = torch.softmax(pred, dim=1)
        labels_pred = torch.argmax(pred_phase, dim=1).cpu().numpy()
        cm = ConfusionMatrix(
            actual_vector=true_label,
            predict_vector=labels_pred,
        )
        return cm.Overall_ACC, cm.PPV, cm.TPR, cm.classes, cm.F1_Macro

    def save_to_drive(self, vid_index):
        acc, ppv, tpr, keys, f1 = self.get_phase_acc(self.current_phase_labels,
                                                     self.current_p_phases)
        save_path = self.pickle_path / f"{self.hparams.fps_sampling_test}fps"
        save_path.mkdir(exist_ok=True)
        save_path_txt = save_path / f"video_{vid_index}_{self.hparams.fps_sampling_test}fps_acc.txt"
        save_path_vid = save_path / f"video_{vid_index}_{self.hparams.fps_sampling_test}fps.pkl"

        with open(save_path_txt, "w") as f:
            f.write(
                f"vid: {vid_index}; acc: {acc}; ppv: {ppv}; tpr: {tpr}; keys: {keys}; f1: {f1}"
            )
            self.test_acc_per_video[vid_index] = acc
            print(
                f"save video {vid_index} | acc: {acc:.4f} | f1: {f1}"
            )
        with open(save_path_vid, 'wb') as f:
            pickle.dump([
                np.asarray(self.current_stems),
                np.asarray(self.current_p_phases),
                np.asarray(self.current_phase_labels)
            ], f)

    def test_step(self, batch, batch_idx):

        x, y_phase, (vid_idx, img_name, img_index, tool_Grasper, tool_Bipolar,
               tool_Hook, tool_Scissors, tool_Clipper, tool_Irrigator,
               tool_SpecimenBag) = batch
        vid_idx_raw = vid_idx.cpu().numpy()
        with torch.no_grad():
            stem, y_hat, _ = self.forward(x)
        self.test_acc_phase(y_hat, y_phase)
        #self.log("test_acc_phase", self.test_acc_phase, on_epoch=True, on_step=True)
        vid_idxs, indexes = np.unique(vid_idx_raw, return_index=True)
        vid_idxs = [int(x) for x in vid_idxs]
        index_next = len(vid_idx) if len(vid_idxs) == 1 else indexes[1]
        for i in range(len(vid_idxs)):
            vid_idx = vid_idxs[i]
            index = indexes[i]
            if vid_idx != self.current_video_idx:
                self.save_to_drive(self.current_video_idx)
                self.current_stems = []
                self.current_phase_labels = []
                self.current_p_phases = []
                if len(vid_idxs) <= i + 1:
                    index_next = len(vid_idx_raw)
                else:
                    index_next = indexes[i+1]  # for the unlikely case that we have 3 phases in one batch
                self.current_video_idx = vid_idx
            y_hat_numpy = np.asarray(y_hat.cpu()).squeeze()
            self.current_p_phases.extend(
                np.asarray(y_hat_numpy[index:index_next, :]).tolist())
            self.current_stems.extend(
                stem[index:index_next, :].cpu().detach().numpy().tolist())
            y_phase_numpy = y_phase.cpu().numpy()
            self.current_phase_labels.extend(
                np.asarray(y_phase_numpy[index:index_next]).tolist())

        if (batch_idx + 1) * self.hparams.batch_size >= self.len_test_data:
            self.save_to_drive(vid_idx)
            print(f"Finished extracting all videos...")

    def validation_epoch_end(self, outputs):
        """
        outputs: list of dicts returned by validation_step, each containing 'val_loss'
        """
        # 1) gather all the per‐step losses
        val_losses = [out["val_loss"] for out in outputs]
        # 2) compute mean
        avg_loss = torch.stack(val_losses).mean()
        # 3) return it so EarlyStopping can see 'val_loss'
        return {
            "val_loss": avg_loss,
            "log": {"val_loss": avg_loss}
        }

    def test_epoch_end(self, outputs):
        train_val_acc = np.mean(
            np.asarray([self.test_acc_per_video[x] for x in self.dataset.vids_for_training])
        )
        val_val_acc = np.mean(
            np.asarray([self.test_acc_per_video[x] for x in self.dataset.vids_for_val])
        )
        test_val_acc = np.mean(
            np.asarray([self.test_acc_per_video[x] for x in self.dataset.vids_for_test])
        )
        overall_acc = float(self.test_acc_phase.compute())

        step = getattr(self, "current_epoch", 0)

        # 3) Handle case where self.logger might be a list of loggers
        if self.logger is not None:
            # Normalize self.logger to a list
            loggers = self.logger if isinstance(self.logger, list) else [self.logger]

            for logger in loggers:
                # 3a) Try TensorBoard‐style logging: .experiment.add_scalar(...)
                try:
                    if hasattr(logger, "experiment") and hasattr(logger.experiment, "add_scalar"):
                        logger.experiment.add_scalar("test_acc_train", train_val_acc, step)
                        logger.experiment.add_scalar("test_acc_val",   val_val_acc,   step)
                        logger.experiment.add_scalar("test_acc_test",  test_val_acc,  step)
                        logger.experiment.add_scalar("test_acc",       overall_acc,   step)
                        # If this succeeds, move on to next logger in the list
                        continue
                except Exception:
                    # Fall through to next attempt (e.g. WandB) if any error occurs
                    pass

                # 3b) Try WandB‐style logging: .experiment.log({...}, step=step)
                try:
                    if hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
                        logger.experiment.log({
                            "test_acc_train": train_val_acc,
                            "test_acc_val":   val_val_acc,
                            "test_acc_test":  test_val_acc,
                            "test_acc":       overall_acc
                        }, step=step)
                        continue
                except Exception:
                    # If that also fails, we simply move on
                    pass

        # 4) Always print to stdout as a fallback / visibility
        print(
            f"[Test] train_acc: {train_val_acc:.4f}, "
            f"val_acc: {val_val_acc:.4f}, "
            f"test_acc: {test_val_acc:.4f}, "
            f"overall_acc: {overall_acc:.4f}"
        )


    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate)
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer]  #, [scheduler]

    def __dataloader(self, split=None):
        dataset = self.dataset.data[split]
        if self.hparams.batch_size > self.hparams.model_specific_batch_size_max:
            print(
                f"The choosen batchsize is too large for this model."
                f" It got automatically reduced from: {self.hparams.batch_size} to {self.hparams.model_specific_batch_size_max}"
            )
            self.hparams.batch_size = self.hparams.model_specific_batch_size_max

        if split == "val" or split == "test":
            should_shuffle = False
        else:
            should_shuffle = True
        print(f"split: {split} - shuffle: {should_shuffle}")
        worker = self.hparams.num_workers
        if split == "test":
            print(
                "worker set to 0 due to test"
            )  # otherwise for extraction the order in which data is loaded is not sorted e.g. 1,2,3,4, --> 1,5,3,2
            worker = 0

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=should_shuffle,
            num_workers=worker,
            pin_memory=True,
        )
        return loader

    def train_dataloader(self):
        """
        Intialize train dataloader
        :return: train loader
        """
        dataloader = self.__dataloader(split="train")
        logging.info("training data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    def val_dataloader(self):
        """
        Initialize val loader
        :return: validation loader
        """
        dataloader = self.__dataloader(split="val")
        logging.info("validation data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader


    def test_dataloader(self):
        dataloader = self.__dataloader(split="test")
        logging.info("test data loader called  - size: {}".format(
            len(dataloader.dataset)))
        print(f"starting video idx for testing: {self.current_video_idx}")
        self.set_export_pickle_path()
        return dataloader

    @staticmethod
    def add_module_specific_args(parser):  # pragma: no cover
        cholec_fe_module = parser.add_argument_group(
            title='cholec_fe_module specific args options')
        cholec_fe_module.add_argument("--learning_rate",
                                      default=0.001,
                                      type=float)
        cholec_fe_module.add_argument("--num_tasks",
                                      default=1,
                                      type=int,
                                      choices=[1, 2])
        cholec_fe_module.add_argument("--optimizer_name",
                                      default="adam",
                                      type=str)
        cholec_fe_module.add_argument("--batch_size", default=32, type=int)
        return parser
