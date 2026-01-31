import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

FIGSIZE = (12, 8)

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

df = pd.read_csv(args.input)

# kingdom distribution
plt.figure(figsize=FIGSIZE)
df["kingdom"].value_counts().plot(kind="bar")
plt.title("Kingdom Counts")
plt.ylabel("Count")
plt.xlabel("Kingdom")
plt.savefig(os.path.join(args.output, "kingdom_distribution.png"))

# length distribution
df["length"] = df["sequence"].apply(lambda seq: len(seq))
plt.figure(figsize=FIGSIZE)
plt.hist(df["length"], bins=50)
plt.yscale("log")
plt.title("Sequence Length Distribution")
plt.ylabel("Count (log scale)")
plt.xlabel("Length")
plt.savefig(os.path.join(args.output, "length_distribution.png"))

# per kingdom length distribution boxplot
plt.figure(figsize=FIGSIZE)
plt.boxplot([df[df["kingdom"] == kingdom]["length"] for kingdom in df["kingdom"].unique()], tick_labels=[kingdom for kingdom in df["kingdom"].unique()])
plt.title("Length Distribution per Kingdom")
plt.ylabel("Length")
plt.xlabel("Kingdom")
plt.savefig(os.path.join(args.output, "length_distribution_per_kingdom.png"))

# per kingdom type distribution
plt.figure(figsize=FIGSIZE)
fig, ax = plt.subplots(2, 2, figsize=FIGSIZE)
ax = ax.flatten()
sp_types = df["sp_type"].unique()
for kingdom, a in zip(df["kingdom"].unique(), ax):
    a.set_title(kingdom)
    bars = a.bar(sp_types, [len(df[(df["kingdom"] == kingdom) & (df["sp_type"] == sp_type)]) for sp_type in sp_types])
    for bar in bars:
        a.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), bar.get_height(), ha="center", va="bottom")
    a.set_ylabel("Count")
    a.set_xlabel("SP Type")

plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.savefig(os.path.join(args.output, "type_distribution_per_kingdom.png"))

# per type length distribution
plt.figure(figsize=FIGSIZE)
plt.boxplot([df[df["sp_type"] == sp_type]["length"] for sp_type in sp_types], tick_labels=sp_types)
plt.title("Length Distribution per SP Type")
plt.ylabel("Length")
plt.xlabel("SP Type")
plt.savefig(os.path.join(args.output, "length_distribution_per_sp_type.png"))

print("Saved all plots to", args.output)