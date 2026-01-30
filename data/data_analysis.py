import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("filepath")
args = parser.parse_args()

data = []
with open(args.filepath, "r") as f:
    c = 0
    while f.readable():
        line = f.readline().strip()
        if line.startswith(">"):
            entry_id, kingdom, sp_type = line.split("|")
            sequence = f.readline().strip()
            annotation = f.readline().strip()
            data.append((entry_id[1:], kingdom, sp_type, sequence, annotation))
        else:
            break

df = pd.DataFrame(data=data, columns=["entry_id", "kingdom", "sp_type", "sequence", "annotation"])
df.to_csv(f"{args.filepath}.csv", index=False)

print(f"Saved {len(df)} to {args.filepath}.csv")

print("Kingdom Counts")
print(df["kingdom"].value_counts())

df["kingdom"].value_counts().plot(kind="bar", figsize=(12, 8))
plt.xlabel("Kingdom")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig("image.png")