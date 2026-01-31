import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
args = parser.parse_args()

def transition_boundaries(sequence):
    # stores index of residue before boundary and transition from, to
    boundaries = []
    for i, letter in enumerate(sequence[:-1]):
        if letter != sequence[i+1]:
            boundaries.append((i, letter, sequence[i+1]))

    return boundaries

df = pd.read_csv(args.input)
df["transition_boundaries"] = df["annotation"].apply(transition_boundaries)
df.to_csv(args.output, index=False)
