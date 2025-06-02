import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np


def write_pkl(root_dir):
    print("writing_pkl...")
    root_dir = Path(root_dir)
    #img_base_path = root_dir / "cholec_split_250px_25fps"
    img_base_path = root_dir / "input"
    annot_tool_path = root_dir / "phase_annotations" /"tool_annotations"
    annot_timephase_path = root_dir / "phase_annotations"
    ##out_path = "Videos/input/cholec80/dataframes/"
    ##out_path.mkdir(parents=True, exist_ok=True)   
    out_path = Path("Videos/input/cholec80/dataframes")
    out_path.mkdir(parents=True, exist_ok=True)

    #print(img_base_path, annot_tool_path, annot_timephase_path, out_path)

    class_labels = [
        "Preparation",
        "CalotTriangleDissection",
        "ClippingCutting",
        "GallbladderDissection",
        "GallbladderPackaging",
        "CleaningCoagulation",
        "GallbladderRetraction",
    ]

    cholec_df = pd.DataFrame(columns=[
        "image_path", "class", "time", "video_idx", "tool_Grasper",
        "tool_Bipolar", "tool_Hook", "tool_Scissors", "tool_Clipper",
        "tool_Irrigator", "tool_SpecimenBag"
    ])
    for i in tqdm(range(1, 81)):
        vid_df = pd.DataFrame()
        #img_path_for_vid = img_base_path / f"video{i:02d}"
        #img_list = sorted(img_path_for_vid.glob('*.png'))
        #img_list = [str(i.relative_to(img_base_path)) for i in img_list]
        img_path_for_vid = img_base_path / f"video{i:02d}"
        img_list = sorted(img_path_for_vid.glob("*.png"))
        # store paths relative to the parent of root_dir
        img_list = [str(p.relative_to(root_dir.parent)) for p in img_list]
        
        vid_df["image_path"] = img_list
        vid_df["video_idx"] = [i] * len(img_list)
        # add image class
        vid_time_and_phase = annot_timephase_path / f"video{i:02d}-phase.txt"
        phases = pd.read_csv(vid_time_and_phase, sep='\t')
        for j, p in enumerate(class_labels):
            phases["Phase"] = phases.Phase.replace({p: j})

        vid_tools = annot_tool_path / f"video{i:02d}-tool.txt"
        tools_short = pd.read_csv(vid_tools, sep='\t')
        tools_df = []
        for row in tools_short.itertuples(index=False):
            numbers_of_repetitions = 25
            tools_df.extend([list(row)] * numbers_of_repetitions)
        tools_df.append(tools_df[-1])
        tools_df = np.array(tools_df)

        tools_df = pd.DataFrame(tools_df[:, 1:],
                                columns=[
                                    "tool_Grasper", "tool_Bipolar",
                                    "tool_Hook", "tool_Scissors",
                                    "tool_Clipper", "tool_Irrigator",
                                    "tool_SpecimenBag"
                                ])

        vid_df = pd.concat([vid_df, phases], axis=1)
        vid_df = pd.concat([vid_df, tools_df], axis=1)
        
        if len(img_list) != vid_df.shape[0]:
            print(f"⚠ MISMATCH for video {i}: "
                  f"images={len(img_list)}, dataframe_rows={vid_df.shape[0]}, "
                  f"tools={len(tools_df)}, phases={len(phases)}")
        else:
        # (optional) print a “✓” for sanity
            print(f" video {i} OK.")
            print(f"len(img_list): {len(img_list)} - vid_df.shape[0]:{vid_df.shape[0]} - len(tools_df):{len(tools_df)} - len(phases):{len(phases)}")
        #print(
        #    f"len(img_list): {len(img_list)} - vid_df.shape[0]:{vid_df.shape[0]} - len(tools_df):{len(tools_df)} - len(phases):{len(phases)}")
        vid_df = vid_df.rename(columns={
            "Phase": "class",
            "Frame": "time",
        })
        cholec_df = cholec_df.append(vid_df, ignore_index=True, sort=False)

    print("DONE")
    print(cholec_df.shape)
    print(cholec_df.columns)

    cholec_df.to_pickle(out_path / "cholec_split_250px_25fps.pkl")
    print("File saved in: ", out_path)


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    root_dir = "Videos/"
    write_pkl(root_dir)
