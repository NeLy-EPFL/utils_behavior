import os
import shutil

Slower = [
    "Empty-Gal4",
    "LC25",
    "LC4",
    "86666 (LH1139)",
    "86637 (LH2220)",
    "86705 (LH1668)",
    "86699 (LH123)",
    "SS52577-gal4 (PBG2‐9.s‐FBℓ3.b‐NO2V.b (PB))",
    "SS00078-gal4 (PBG2‐9.s‐FBℓ3.b‐NO2D.b (PB))",
    "SS02239-gal4 (P-F3LC patch line)",
    "MB312B (PAM-07)",
    "MB043B (PAM-11)",
    "MB315C (PAM-01)",
    "MB504B (All PPL1)",
    "MB063B (PAM-10)",
]

Faster = [
    "LC10-2",
    "PR",
    "R15B07-gal4 (R1/R3/R4d (EB))",
    "R78A01 (ExR1 (EB))",
    "R59B10-gal4 (R4m - medial (EB))",
    "R38H02-gal4 (R4 (EB))",
    "c316 (MB-DPM)",
    "51975 (Tk-GAL4 5Fa)",
    "51976 (Crz-GAL4 3M)",
    "51977 (Crz-GAL4 4M)",
    "51978 (AstA-GAL4 3M)",
    "51979 (AstA-GAL4 5)",
    "51970 (Capa-GAL4 5F)",
    "51988 (Dh31-GAL4 2M)",
    "51985 (Ms-GAL4 1M)",
    "51986 (Ms-GAL4 6Ma)",
    "25681 (NPF-GAL4 2)",
    "25682 (NPF-GAL4 1)",
    "51980 (Burs-GAL4 4M)",
]

Saturate = [
    "LC16-1",
    "LC6",
    "LC12",
    "LPLC2",
    "75823 (LH2446)",
    "75945 (LH1543)",
    "86676 (LH191)",
    "86630 (LH2385)",
    "86632 (LH2392)",
    "86681 (LH141)",
    "86674 (LH247)",
    "86667 (LH1000)",
    "86682 (LH85)",
    "86639 (LH1990)",
    "86685 (LH272)",
    "86707 (LH578)",
    "86671 (LH412)",
    "R19G02 (E-PG)",
    "SS32219-Gal4 (LAL-2)",
    "SS32230-Gal4 (LAL-1)",
    "MB113C (OA-VPM4)",
    "MB296B (PPL1-03)",
    "MB399B",
    "MB434B",
    "MB418B ",
    "MBON-13-GaL4 (MBON-α′2)",
    "MBON-01-GaL4 (MBON-γ5β′2a)",
]

NoFinish = [
    "34497 (MZ19-GAL4)",
    "DDC-gal4",
    "Ple-Gal4.F a.k.a TH-Gal4",
    "MB504B (All PPL1)",
    "VT43924 (MB-APL)",
    "41744 (IR8a mutant)",
    "50742 (MB247-GAL4)",
    "MB247-Gal4",
    "854 (OK107-Gal4)",
]

Videopath = "/mnt/upramdya_data/MD/TNT_Screen_RawGrids"
outpath = "/mnt/upramdya_data/MD/TNT_Hits_Grids"

# Create directories for each category if they don't exist
categories = {
    "Slower": Slower,
    "Faster": Faster,
    "Saturate": Saturate,
    "NoFinish": NoFinish,
}

for category_name in categories.keys():
    dir_path = os.path.join(outpath, category_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Copy videos to corresponding directories
for category_name, category_list in categories.items():
    for name in category_list:
        for file in os.listdir(Videopath):
            if name in file:
                src_file = os.path.join(Videopath, file)
                dest_file = os.path.join(outpath, category_name, file)
                if not os.path.exists(dest_file):
                    shutil.copy(src_file, dest_file)
                    print(f"Copied {file} to {dest_file}")
                else:
                    print(f"File {file} already exists in {dest_file}")
