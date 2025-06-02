import re
from pathlib import Path

def pad_cholec80_pickles(base_export_dir):
    """
    Walks through each <fps>fps subfolder under base_export_dir, finds any
    'video_<N>_<fps>fps.pkl' where N is a single digit, and renames it to
    'video_0<N>_<fps>fps.pkl'.
    """
    base = Path(base_export_dir)
    if not base.exists():
        raise ValueError(f"{base_export_dir} does not exist")

    # pattern to match single‐digit filenames like "video_8_1.0fps.pkl"
    single_digit_re = re.compile(r"^video_(\d)_(\d+\.\d+)fps\.pkl$")
    # iterate over all subdirectories named "*fps"
    for fps_folder in base.iterdir():
        if not fps_folder.is_dir() or not fps_folder.name.endswith("fps"):
            continue

        for pkl in fps_folder.iterdir():
            m = single_digit_re.match(pkl.name)
            if m:
                digit, fps_str = m.groups()
                old_name = pkl
                new_name = fps_folder / f"video_{int(digit):02d}_{fps_str}fps.pkl"
                print(f"Renaming {old_name.name} → {new_name.name}")
                old_name.rename(new_name)


if __name__ == "__main__":
    # Adjust this path to wherever your cholec80_pickle_export folder is
    # For example: "/home/jovyan/TeCNO/logs/.../cholec80_pickle_export"
    export_dir = "logs/250602-110750_FeatureExtraction_Cholec80FeatureExtract_cnn_TwoHeadResNet50Model/cholec80_pickle_export"
    pad_cholec80_pickles(export_dir)
