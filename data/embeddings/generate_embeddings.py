import pandas as pd
import argparse
import os
import torch
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm
import re
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("type", choices=["per_protein", "per_residue"])
parser.add_argument("--save_steps", default=-1, type=int, required=False) # use -1 as negative value to never save in between
parser.add_argument("--batch_size", default=8, type=int, required=False)
parser.add_argument("--save", )
args = parser.parse_args()

PER_PROTEIN = args.type == "per_protein"
BATCH_SIZE = args.batch_size
SAVE_STEPS = args.save_steps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE}")

os.makedirs(os.path.join(args.output, "embeddings", args.type), exist_ok=True)

print("Loading ProtT5...")
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', dtype=torch.float16)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", dtype=torch.float16)
model.to(DEVICE)
model.eval()
print("ProtT5 loaded!")

print("Loading data...")
df = pd.read_csv(args.input)
print("Spacing Sequences...")
df["spaced_sequence"] = df["sequence"].apply(lambda seq: " ".join(list(re.sub(r"[UZOB]", "X", seq))))

print("Generating Embeddings...")
embeddings = []
for i in tqdm(range(0, len(df), BATCH_SIZE), total=len(df) // BATCH_SIZE):
    batch = [(row["entry_id"], row["spaced_sequence"]) for _, row in df.iloc[i:i+BATCH_SIZE].iterrows()]

    ids = tokenizer([seq for _, seq in batch], add_special_tokens=True, padding="longest", return_tensors="pt")
    input_ids = ids["input_ids"].to(DEVICE)
    attention_masks = ids["attention_mask"].to(DEVICE)

    with torch.no_grad():
        batch_embeddings = model(input_ids=input_ids, attention_mask=attention_masks).last_hidden_state[:, :-1, :]

    if PER_PROTEIN:
        # match shape of embeddings --> multiplication
        attention_masks = attention_masks[:, :-1].unsqueeze(-1).expand(batch_embeddings.size())
        summed_batch_embeddings = torch.sum(batch_embeddings * attention_masks, dim=1)
        batch_lengths = torch.sum(attention_masks, dim=1)
        prot_batch_embeddings = summed_batch_embeddings / batch_lengths
        embeddings.extend([(acc_id, prot_embedding) for (acc_id, _), prot_embedding in zip(batch, prot_batch_embeddings)])

    else:
        attention_masks = attention_masks[:, :-1]
        embeddings.extend([(acc_id, residue_embeddings, attention_mask) for (acc_id, _), residue_embeddings, attention_mask in zip(batch, batch_embeddings, attention_masks)])

    if SAVE_STEPS != -1 and i % SAVE_STEPS == 0:
        with open(os.path.join(args.output, "embeddings", args.type, f"step_{i}.pkl"), "wb") as f:
            pkl.dump(embeddings, f)
        print(f"Saved at step {i}")

with open(os.path.join(args.output, "embeddings", args.type, "final_embeddings.pkl"), "wb") as f:
    pkl.dump(embeddings, f)
print(f"Saved {len(embeddings)} embeddings after final step")