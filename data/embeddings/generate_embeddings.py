import pandas as pd
import argparse
import os
import torch
from transformers import T5Tokenizer, T5EncoderModel, AutoModelForSeq2SeqLM
from tqdm import tqdm
import re
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("type", choices=["per_protein", "per_residue"])
parser.add_argument("--save_steps", default=-1, type=int, required=False) # use -1 as negative value to never save in between
parser.add_argument("--batch_size", default=8, type=int, required=False)
parser.add_argument("--model", choices=["prott5", "prostt5"], default="prott5", required=False)
parser.add_argument("--prev_savepoint", default=-1, required=False)
args = parser.parse_args()

PER_PROTEIN = args.type == "per_protein"
MODEL = args.model
BATCH_SIZE = args.batch_size
SAVE_STEPS = args.save_steps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE}")

# taken from: https://github.com/mheinzinger/ProstT5
gen_kwargs_aa2fold = {
                  "do_sample": True,
                  "num_beams": 3,
                  "top_p" : 0.95,
                  "temperature" : 1.2,
                  "top_k" : 6,
                  "repetition_penalty" : 1.2,
}

os.makedirs(os.path.join(args.output, f"{MODEL}_embeddings", args.type), exist_ok=True)

if MODEL == "prott5":
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

    # load previous savepoint
    if args.prev_savepoint != -1:
        print(f"Loading from {args.prev_savepoint}")
        with open(os.path.join(args.prev_savepoint), "rb") as f:
            embeddings = pkl.load(f)

    completed_ids = set([entry[0] for entry in embeddings])

    for i in tqdm(range(0, len(df), BATCH_SIZE), total=len(df) // BATCH_SIZE):
        batch = [(row["entry_id"], row["spaced_sequence"]) for _, row in df.iloc[i:i+BATCH_SIZE].iterrows() if row["entry_id"] not in completed_ids]
        if len(batch) == 0:
            continue

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
            with open(os.path.join(args.output, f"{MODEL}_embeddings", args.type, f"step_{i}.pkl"), "wb") as f:
                pkl.dump(embeddings, f)
            print(f"Saved at step {i}")

    with open(os.path.join(args.output, f"{MODEL}_embeddings", args.type, "final_embeddings.pkl"), "wb") as f:
        pkl.dump(embeddings, f)
    print(f"Saved {len(embeddings)} embeddings after final step")

elif MODEL == "prostt5":
    print("Loading ProstT5...")
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)
    embedding_model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(DEVICE)
    translation_model = AutoModelForSeq2SeqLM.from_pretrained("Rostlab/ProstT5").to(DEVICE)

    # half only supported on gpu
    embedding_model.float() if DEVICE == "cpu" else embedding_model.half()
    translation_model.float() if DEVICE == "cpu" else translation_model.half()
    print(f"ProstT5 Loaded to {DEVICE}!")

    print("Loading data...")
    df = pd.read_csv(args.input)
    df["spaced_sequence"] = df["sequence"].apply(lambda seq: "<AA2fold> " + " ".join(list(re.sub(r"[UZOB]", "X", seq))))

    print("Generating batches...")
    embeddings = []

    # load previous savepoint
    if args.prev_savepoint != -1:
        print(f"Loading from {args.prev_savepoint}")
        with open(os.path.join(args.prev_savepoint), "rb") as f:
            embeddings = pkl.load(f)

    completed_ids = set([entry[0] for entry in embeddings])

    for i in tqdm(range(0, len(df), BATCH_SIZE), total=len(df) // BATCH_SIZE):
        batch = [(row["entry_id"], row["sequence"], row["spaced_sequence"]) for _, row in df.iloc[i:i+BATCH_SIZE].iterrows() if row["entry_id"] not in completed_ids]
        if len(batch) == 0:
            continue

        max_len = max(len(s) for _, s, _ in batch)
        min_len = min(len(s) for _, s, _ in batch)

        # 3Di sequence generation
        batch_aa_ids = tokenizer([s for _, _, s in batch], add_special_tokens=True, padding="longest", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            batch_translations = translation_model.generate(
                input_ids=batch_aa_ids["input_ids"],
                attention_mask=batch_aa_ids["attention_mask"],
                max_length=max_len,
                min_length=min_len,
                early_stopping=True,
                num_return_sequences=1,
                **gen_kwargs_aa2fold
            )

        batch_decoded_translations = tokenizer.batch_decode(batch_translations, skip_special_tokens=True)
        batch_structure_sequences = ["".join(ts.split(" ")) for ts in batch_decoded_translations]
        batch = [(entry_id, sequence, spaced_sequence, di_sequence) for (entry_id, sequence, spaced_sequence), di_sequence in zip(batch, batch_structure_sequences)]

        # AA seq + 3Di sequence generation
        batch_input_3di_seqs = ["<fold2AA> " + " ".join(list(s)) for _, _, _, s in batch]
        batch_3di_ids = tokenizer(batch_input_3di_seqs, add_special_tokens=True, padding="longest", return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            batch_aa_embeddings = embedding_model(batch_aa_ids["input_ids"], attention_mask=batch_aa_ids["attention_mask"]).last_hidden_state[:, 1:-1]
            batch_3di_embeddings = embedding_model(batch_3di_ids["input_ids"], attention_mask=batch_3di_ids["attention_mask"]).last_hidden_state[:, 1:-1]

        if PER_PROTEIN:
            # match shape of embeddings --> multiplication
            batch_aa_attention_masks = batch_aa_ids["attention_mask"][:, 1:-1].unsqueeze(-1).expand(batch_aa_embeddings.size())
            batch_3di_attention_masks = batch_3di_ids["attention_mask"][:, 1:-1].unsqueeze(-1).expand(batch_3di_embeddings.size())

            # sum across sequence lengths
            batch_summed_aa_embeddings = torch.sum(batch_aa_embeddings * batch_aa_attention_masks, dim=1)
            batch_summed_3di_embeddings = torch.sum(batch_3di_embeddings * batch_3di_attention_masks, dim=1)

            # get lengths from attention masks
            batch_aa_lengths = torch.sum(batch_aa_attention_masks, dim=1)
            batch_3di_lengths = torch.sum(batch_3di_attention_masks, dim=1)

            # calculate mean
            batch_prot_aa_embeddings = batch_summed_aa_embeddings / batch_aa_lengths
            batch_prot_3di_embeddings = batch_summed_3di_embeddings / batch_3di_lengths

            # store
            embeddings.extend([(acc_id, prot_aa_embedding, prot_3di_embedding) for (acc_id, _, _, _), prot_aa_embedding, prot_3di_embedding in zip(batch, batch_prot_aa_embeddings, batch_prot_3di_embeddings)])

        else:
            batch_aa_attention_masks = batch_aa_ids["attention_mask"][:, 1:-1]
            batch_3di_attention_masks = batch_3di_ids["attention_mask"][:, 1:-1]
            embeddings.extend([(acc_id, residue_aa_embeddings, residue_aa_attention_mask, residue_3di_embeddings, residue_3di_attention_mask) for (acc_id, _, _, _), residue_aa_embeddings, residue_aa_attention_mask, residue_3di_embeddings, residue_3di_attention_mask in zip(batch, batch_aa_embeddings, batch_aa_attention_masks, batch_3di_embeddings, batch_3di_attention_masks)])

        if SAVE_STEPS != -1 and i % SAVE_STEPS == 0:
            with open(os.path.join(args.output, f"{MODEL}_embeddings", args.type, f"step_{i}.pkl"), "wb") as f:
                pkl.dump(embeddings, f)
            print(f"Saved at step {i}")

    with open(os.path.join(args.output, f"{MODEL}_embeddings", args.type, "final_embeddings.pkl"), "wb") as f:
        pkl.dump(embeddings, f)
    print(f"Saved {len(embeddings)} embeddings after final step")

else:
    print(f"Unknown Model: {MODEL}\nExiting!")