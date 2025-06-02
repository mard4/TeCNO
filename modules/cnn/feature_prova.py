# modules/cnn/feature_prova.py

import torch
import numpy as np
from argparse import Namespace

# Import the original FeatureExtraction from feature_extraction.py
from feature_extraction import FeatureExtraction

# ─── DummyLogger / DummySummaryWriter ────────────────────────────────────────

class DummySummaryWriter:
    def add_scalar(self, tag, scalar_value, global_step=None):
        print(f"Logged (TensorBoard) → {tag}: {scalar_value}  (step={global_step})")

    # If you also rely on WandB's `.log({...}, step=…)`, you can do:
    def log(self, data_dict, step=None):
        for k, v in data_dict.items():
            print(f"Logged (WandB) → {k}: {v}  (step={step})")

class DummyLogger:
    def __init__(self):
        # Emulate Lightning’s TensorBoardLogger or WandbLogger:
        self.experiment = DummySummaryWriter()


# ─── DummyDataset ─────────────────────────────────────────────────────────────

class DummyDataset:
    def __init__(self, num_classes=7):
        # These IDs are used in test_epoch_end when computing means:
        self.vids_for_training = [101, 202]
        self.vids_for_val      = [303]
        self.vids_for_test     = [404, 505]

        # test_epoch_end references dataset.df["test"].video_idx.min()
        fake_video_idx = Namespace(min=lambda: 999)
        self.df = { "test": Namespace(video_idx=fake_video_idx) }

        # test_dataloader looks at len(self.dataset.data["test"])
        self.data = { "test": [0, 1, 2] }

        # *** Crucial: FeatureExtraction.__init__ needs this to exist: ***
        # Provide a simple all-ones weight vector of length=num_classes:
        self.class_weights = np.ones((num_classes,), dtype=np.float32)


# ─── FakeFeatureExtraction ────────────────────────────────────────────────────

class FakeFeatureExtraction(FeatureExtraction):
    def __init__(self):
        # 1) Build a minimal hparams Namespace so that FeatureExtraction.__init__ runs
        hparams = Namespace(
            batch_size=1,
            num_tasks=1,
            learning_rate=1e-3,
            out_features=7,          # must match DummyDataset.num_classes
            features_per_seconds=25,  # not used in our test, but needed to avoid missing attributes
            features_subsampling=5,
            fps_sampling=1,
            fps_sampling_test=1,
            input_height=224,
            input_width=224,
            num_workers=0,
            data_root=".",           # irrelevant here
        )

        # 2) Create a DummyDataset with class_weights defined
        dummy_ds = DummyDataset(num_classes=hparams.out_features)

        # 3) Call FeatureExtraction.__init__ with (hparams, model=None, dataset=dummy_ds)
        super().__init__(hparams=hparams, model=None, dataset=dummy_ds)

        # 4) Overwrite self.dataset just to be explicit (though super() already set it)
        self.dataset = dummy_ds

        # 5) Provide a dummy test_acc_per_video dictionary
        #    test_epoch_end will do: 
        #      np.mean([self.test_acc_per_video[x] for x in self.dataset.vids_for_training]), etc.
        self.test_acc_per_video = {
            101: 0.80,
            202: 0.60,
            303: 0.50,
            404: 0.90,
            505: 0.75
        }

        # 6) Make a fake metric that inherits from nn.Module so assignment is valid:
        class DummyMetric(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def compute(self):
                # Return a torch.Tensor with the “overall” value
                return torch.tensor(0.82)

        self.test_acc_phase = DummyMetric()

        # 7) Give the module a logger that has .experiment.add_scalar(...)
        self.logger = DummyLogger()

        # 8) Fake “current_epoch” so that add_scalar(step=…) has a value
        self.current_epoch = 42

    # We do not override test_epoch_end; use the patched version in feature_extraction.py


if __name__ == "__main__":
    print("➤ Running FakeFeatureExtraction.test_epoch_end(...)")

    module = FakeFeatureExtraction()
    # We pass an empty list for `outputs` since our patched test_epoch_end
    # only uses self.test_acc_per_video and self.test_acc_phase, not the outputs arg.
    module.test_epoch_end(outputs=[])

    print("✅ test_epoch_end completed without crashing.")
