#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

def write_pkl(root_dir):
    print("writing_pkl...")
    root_dir = Path(root_dir)
    img_base_path = root_dir / "input"
    annot_tool_path = root_dir / "phase_annotations" /"tool_annotations"
    annot_timephase_path = root_dir / "phase_annotations"
    out_path = Path("Videos/input/cholec80/dataframes")
    out_path.mkdir(parents=True, exist_ok=True)

    class_labels = [
        "Preparation",
        "CalotTriangleDissection",
        "ClippingCutting",
        "GallbladderDissection",
        "GallbladderPackaging",
        "CleaningCoagulation",
        "GallbladderRetraction",
    ]

    # master DataFrame
    cholec_df = pd.DataFrame(columns=[
        "image_path", "class", "time", "video_idx", "tool_Grasper",
        "tool_Bipolar", "tool_Hook", "tool_Scissors", "tool_Clipper",
        "tool_Irrigator", "tool_SpecimenBag"
    ])

    for vid_idx in tqdm(range(1, 81), desc="Building DataFrame"):
        # 1) Gather images
        img_folder = img_base_path / f"video{vid_idx:02d}"
        img_list = sorted(img_folder.glob("*.png"))
        # store paths relative to img_base_path
        img_paths = [str(p.relative_to(img_base_path)) for p in img_list]
        N_images = len(img_paths)

        # 2) Load & encode phases
        phases_file = annot_timephase_path / f"video{vid_idx:02d}-phase.txt"
        phases = pd.read_csv(phases_file, sep="\t")
        # convert phase names to integer labels
        for j, name in enumerate(class_labels):
            phases["Phase"] = phases["Phase"].replace({name: j})

        # 3) Load & repeat tools
        tools_file = annot_tool_path / f"video{vid_idx:02d}-tool.txt"
        tools_short = pd.read_csv(tools_file, sep="\t")
        tools_rows = []
        for row in tools_short.itertuples(index=False):
            tools_rows.extend([list(row)[1:]] * 25)  # 25 fps → one row per frame
        # if one extra timestamp row gets added upstream, we'll trim or pad below
        tools_df = pd.DataFrame(
            tools_rows,
            columns=[
                "tool_Grasper", "tool_Bipolar", "tool_Hook",
                "tool_Scissors", "tool_Clipper", "tool_Irrigator",
                "tool_SpecimenBag"
            ]
        )

        # 4) Fix any length mismatches
        def align(df, name):
            L = len(df)
            if L > N_images:
                # trim extra rows at end
                df = df.iloc[:N_images].reset_index(drop=True)
            elif L < N_images:
                # pad by repeating last row
                last = df.iloc[-1]
                pad = pd.DataFrame([last] * (N_images - L), columns=df.columns)
                df = pd.concat([df, pad], ignore_index=True)
            return df

        phases = align(phases, "phases")
        tools_df = align(tools_df, "tools")

        # sanity check
        if len(phases) != N_images or len(tools_df) != N_images:
            raise RuntimeError(
                f"Video {vid_idx:02d} still mismatched after align(): "
                f"imgs={N_images}, phases={len(phases)}, tools={len(tools_df)}"
            )

        # 5) Build vid_df
        vid_df = pd.DataFrame({
            "image_path": img_paths,
            "video_idx":  [vid_idx] * N_images,
            "class":      phases["Phase"],
            "time":       phases["Frame"] if "Frame" in phases.columns else phases.iloc[:, 0]
        })

        # append tool columns
        vid_df = pd.concat([vid_df, tools_df], axis=1)

        # 6) Append to master
        cholec_df = pd.concat([cholec_df, vid_df], ignore_index=True, sort=False)

    # 7) Save out
    print("DONE — final shape:", cholec_df.shape)
    cholec_df.to_pickle(out_path / "cholec_split_250px_25fps.pkl")
    print("File saved to:", out_path / "cholec_split_250px_25fps.pkl")


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    write_pkl(out_path)
