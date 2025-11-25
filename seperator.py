import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

src_dir = "D:\\programms\\ML\\cropped_backup"
out_dir = "D:\\programms\\ML\\organised_dataset"

labels = pd.read_csv("C:\\Users\\Remi\\Downloads\\labels.csv")

all_file_paths = os.listdir(src_dir)

prefix_to_files = {}

for f in all_file_paths:
    # extract part before first "_"  
    prefix = f.split("_")[0]
    prefix_to_files.setdefault(prefix, []).append(f)

# Expand dataframe rows into multiple rows (1 per actual file)
rows = []
for _, row in labels.iterrows():
    prefix = str(row["Kood"])
    label = row["Liik"]

    matching_files = prefix_to_files.get(prefix, [])

    for f in matching_files:
        rows.append({
            "prefix": prefix,
            "label": label,
            "filename": f
        })

expanded_df = pd.DataFrame(rows)


# Train/test split (stratified)
train_df, test_df = train_test_split(
    expanded_df, test_size=0.2, stratify=expanded_df["label"], random_state=42
)

# Function to copy files
def move_files(subset_df, subset_name):
    for _, row in subset_df.iterrows():
        label = row["label"]
        filename = row["filename"]

        target_dir = os.path.join(out_dir, subset_name, label)
        os.makedirs(target_dir, exist_ok=True)

        src = os.path.join(src_dir, filename)
        dst = os.path.join(target_dir, filename)
        try:
            shutil.move(src, dst)  # uncomment if you want to move instead
        except FileNotFoundError:
            print(f"File not found: {src}")

# Move them
move_files(train_df, "train")
move_files(test_df, "test")

print("Done! Files organized into train/test by label.")
