'''
Q4: data statistics before and after pre-processing, measured with the google-t5/t5-small tokenizer.

Usage (from part2/):  python data_stats.py

Definitions used in the report:
  * length       = number of T5 tokens per line, without the </s> end-of-sequence token
  * vocabulary   = number of distinct T5 token ids that occur in the split (special tokens excluded)
  * unk lines    = lines containing at least one <unk> token (information the model cannot produce)
'''
import numpy as np
from transformers import T5TokenizerFast

from preprocess import decode_sql, preprocess_nl, preprocess_sql

DATA_FOLDER = "data"
SPLITS = ["train", "dev"]


def load_lines(path):
    # Keep raw lines (only the newline removed) so "before pre-processing" is truly the raw data.
    with open(path, "r") as f:
        return [line.rstrip("\n") for line in f]


def token_stats(tokenizer, texts):
    # Tokenize without special tokens and summarize lengths, vocabulary and <unk> occurrences.
    ids = [tokenizer(text, add_special_tokens=False).input_ids for text in texts]
    lengths = np.array([len(seq) for seq in ids])
    return {
        "mean_len": lengths.mean(),
        "max_len": int(lengths.max()),
        "vocab": len({tok for seq in ids for tok in seq}),
        "unk_lines": sum(tokenizer.unk_token_id in seq for seq in ids),
    }


def main():
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    # Compute statistics for each split, once on raw text and once after pre-processing.
    results = {}
    for split in SPLITS:
        nl = load_lines(f"{DATA_FOLDER}/{split}.nl")
        sql = load_lines(f"{DATA_FOLDER}/{split}.sql")
        assert len(nl) == len(sql), f"{split}: {len(nl)} NL lines vs {len(sql)} SQL lines"
        results[split] = {
            "n": len(nl),
            "before": {"nl": token_stats(tokenizer, nl), "sql": token_stats(tokenizer, sql)},
            "after": {
                "nl": token_stats(tokenizer, [preprocess_nl(x) for x in nl]),
                "sql": token_stats(tokenizer, [preprocess_sql(x) for x in sql]),
            },
        }

    # Print one table per stage, in the row order of Tables 1 and 2 in the report.
    rows = [
        ("Mean sentence length", "nl", "mean_len", "{:.2f}"),
        ("Mean SQL query length", "sql", "mean_len", "{:.2f}"),
        ("Vocabulary size (natural language)", "nl", "vocab", "{}"),
        ("Vocabulary size (SQL)", "sql", "vocab", "{}"),
        ("Max sentence length", "nl", "max_len", "{}"),
        ("Max SQL query length", "sql", "max_len", "{}"),
        ("SQL queries containing <unk>", "sql", "unk_lines", "{}"),
    ]
    # Check that pre-processing is lossless: encode -> decode -> post-process must give back every
    # original query exactly (verifies the "<" placeholder and the decoding settings).
    for split in SPLITS:
        sql = load_lines(f"{DATA_FOLDER}/{split}.sql")
        exact = sum(
            decode_sql(tokenizer, tokenizer(preprocess_sql(q)).input_ids) == q.strip() for q in sql
        )
        print(f"{split}: {exact}/{len(sql)} SQL queries survive the encode/decode round trip exactly")

    for stage in ["before", "after"]:
        print(f"\n=== {stage} pre-processing ===")
        print(f"{'Number of examples':40s}" + "".join(f"{results[s]['n']:>12}" for s in SPLITS))
        for name, field, key, fmt in rows:
            values = [fmt.format(results[s][stage][field][key]) for s in SPLITS]
            print(f"{name:40s}" + "".join(f"{v:>12}" for v in values))


if __name__ == "__main__":
    main()
