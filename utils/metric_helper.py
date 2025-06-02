# modules/metric_helper.py

import torch
import numpy as np
from typing import Any, Optional, Union
from pycm import ConfusionMatrix

"""
We no longer rely on PL’s Precision/Recall (they changed signatures in PL 0.8.5).
Instead, we implement simple micro‐averaged precision and recall here.
"""

class PrecisionOverClasses:
    """
    Computes micro‐precision over all frames:
      precision = (total true positives) / (total predicted positives).
    For multiclass (single‐label per frame), "predicted positives" = total frames, 
    so micro‐precision = accuracy. But we provide this as a separate method 
    so tecno.py can call it in the same way it expects a 'compute()' method.
    """
    def __init__(self, num_classes: int = 1):
        # `num_classes` is not strictly needed for micro‐precision, but we keep it for API consistency.
        self.num_classes = num_classes

    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        preds: tensor of shape [batch_size, T] (each entry is predicted class index)
        target: tensor of shape [batch_size, T] (each entry is true class index)
        Returns a 0‐D tensor containing micro‐precision.
        """
        # Flatten to 1D
        p = preds.view(-1)
        t = target.view(-1)
        # Count where prediction matches target:
        correct = torch.sum(p == t).float()
        total_predicted = p.numel()  # same as total number of frames
        if total_predicted == 0:
            return torch.tensor(0.0)
        return correct / total_predicted

    # For compatibility with tecno.py's calls to `compute()`, we alias:
    compute = __call__


class RecallOverClasse:
    """
    Micro‐recall is the same as micro‐precision for single‐label multiclass,
    because number of actual positives = total frames, number of true positives 
    is same. So recall = (total true positives)/(total actual positives).
    We implement identically to PrecisionOverClasses.
    """
    def __init__(self, num_classes: int = 1):
        self.num_classes = num_classes

    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Flatten to 1D
        p = preds.view(-1)
        t = target.view(-1)
        correct = torch.sum(p == t).float()
        total_actual = t.numel()
        if total_actual == 0:
            return torch.tensor(0.0)
        return correct / total_actual

    compute = __call__


class AccuracyStages:
    """
    Replacement for the previous Lightning‐based Metric:
    Tracks per‐stage accuracy across all frames seen so far.
    Usage in tecno.py remains identical:

      self.train_acc_stages = AccuracyStages(num_stages=N)
      ...
      acc_stages = self.train_acc_stages(y_pred, y_true)
      # acc_stages is a Python list of length N:
      #   [acc_stage1, acc_stage2, ..., acc_stageN]

    Each call to __call__ updates the internal counters and returns the current accuracy list.
    """

    def __init__(self, num_stages: int = 1, *args, **kwargs):
        self.num_stages = num_stages
        self.reset()

    def reset(self):
        """
        Clear all counters. Call at the start of each epoch if needed.
        """
        self.total_frames = 0
        # correct_counts[s] will track total correct frames for stage s
        self.correct_counts = [0] * self.num_stages

    def __call__(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds: tensor of shape [num_stages, batch_size, num_classes, T]
        target: tensor of shape [batch_size, T] (or [batch_size, 1, T])
        Returns a Python list of floats: [acc_stage1, acc_stage2, ..., acc_stageN].
        """
        self.update(preds, target)
        return self.compute()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Accumulate statistics from one batch. We do:

          - total_frames += number of frames in this batch (target.numel()).
          - For each stage s: take argmax over the “class” dimension in preds[s],
            compare to target, and increment correct_counts[s].

        After calling update(...), someone can call compute() to get accuracies.
        """
        # 1) Count how many frames in this batch:
        num_frames_in_batch = target.numel()
        self.total_frames += num_frames_in_batch

        # 2) If target has an extra dimension [batch_size, 1, T], squeeze it to [batch_size, T]:
        if target.dim() == 3 and target.size(1) == 1:
            t = target.squeeze(1)
        else:
            t = target  # shape [batch_size, T]

        for s in range(self.num_stages):
            # preds[s] has shape [batch_size, num_classes, T]
            # We want per‐frame predicted class index → argmax over dim=1 → [batch_size, T]
            preds_logits = preds[s]
            preds_stage = torch.argmax(preds_logits, dim=1)  # [batch_size, T]

            # Check shapes match:
            assert preds_stage.shape == t.shape, (
                f"Stage {s}: preds_stage {preds_stage.shape} vs target {t.shape}"
            )

            # Count correct frames in this batch for stage s:
            correct_in_batch = int(torch.sum(preds_stage == t).item())
            self.correct_counts[s] += correct_in_batch

    def compute(self):
        """
        Return a Python list of stage accuracies:
          [correct_counts[0]/total_frames, correct_counts[1]/total_frames, ...].
        If no frames were seen (total_frames == 0), return all zeros.
        """
        if self.total_frames == 0:
            return [0.0] * self.num_stages
        return [
            self.correct_counts[s] / float(self.total_frames)
            for s in range(self.num_stages)
        ]


def calc_average_over_metric(metric_list, normlist):
    """
    Unchanged from before. A helper to average metrics across videos, etc.
    """
    for i in metric_list:
        metric_list[i] = np.asarray(
            [0 if value == "None" else value for value in metric_list[i]]
        )
        if normlist[i] == 0:
            metric_list[i] = 0
        else:
            metric_list[i] = metric_list[i].sum() / normlist[i]
    return metric_list


def create_print_output(print_dict, space_desc, space_item):
    """
    Unchanged from before. Formats a dict of lists into a printable string.
    """
    msg = ""
    for key, value in print_dict.items():
        msg += f"{key:<{space_desc}}"
        for i in value:
            msg += f"{i:>{space_item}}"
        msg += "\n"
    msg = msg[:-1]
    return msg
