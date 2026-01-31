import pandas as pd
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
args = parser.parse_args()

def cleavage_site(transition_boundaries):
    for transition in transition_boundaries:
        if transition[1] in {"S", "T", "L"}:
            return transition
    
    return None

df = pd.read_csv(args.input)
df["cleavage_site"] = df["transition_boundaries"].apply(ast.literal_eval).apply(cleavage_site)
df.to_csv(args.output, index=False)