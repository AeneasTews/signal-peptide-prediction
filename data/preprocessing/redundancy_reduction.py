import pandas as pd
import argparse
import edlib
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
args = parser.parse_args()

def cluster_greedy(sequences, identity_threshold):
    # sequences: [(id, sequence), ...]
    sequences.sort(key=lambda x: len(x[1]), reverse=True)
    
    representatives = []
    
    for curr_id, curr_seq in tqdm(sequences, total=len(sequences)):
        is_redundant = False
        
        for _, rep_seq in representatives:
            
            # speed optimization on length
            # if length difference is too big to satisfy identity skip alignment
            # max score = len(curr_seq). if len(rep) is much larger identity drops
            min_len = min(len(curr_seq), len(rep_seq))
            max_len = max(len(curr_seq), len(rep_seq))
            
            if min_len / max_len < identity_threshold:
                continue
                
            # calculate edit distance
            result = edlib.align(curr_seq, rep_seq, mode="NW", task="distance")
            edit_distance = result["editDistance"]
            
            # calculate identity (Length - Distance) / Length
            # defined by length of longer sequence
            # CD-HIT def: matches / length of shorter
            similarity = 1 - (edit_distance / max_len)
            
            if similarity >= identity_threshold:
                is_redundant = True
                break
        
        if not is_redundant:
            representatives.append((curr_id, curr_seq))
            
    return representatives


df = pd.read_csv(args.input)

reps = cluster_greedy(list(zip(df["entry_id"], df["sequence"])), identity_threshold=0.3)
print(f"Reduced from {len(df)} to {len(reps)} sequences")

rep_df = pd.DataFrame(reps, columns=["entry_id", "sequence"])
joined_df = rep_df.merge(df.drop(columns=["sequence"]), on="entry_id", how="left")
joined_df.to_csv(args.output, index=False)