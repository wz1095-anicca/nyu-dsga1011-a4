'''
Text pre-/post-processing shared by data statistics, training and inference, so every stage
transforms text the same way.

Why it is needed: the T5 SentencePiece vocabulary has no "<" character, so the tokenizer maps it to
<unk> and "x < 5" / "x <= 5" can never be generated. About 9% of the training SQL queries contain
"<". We therefore replace "<" with a placeholder word before tokenization and restore it after
decoding. The placeholder never occurs in the train/dev SQL, and with this mapping every train/dev
query survives an encode -> decode round trip exactly.
'''

LESS_THAN_PLACEHOLDER = "LESS"


def preprocess_nl(nl):
    # Natural-language input: only strip surrounding whitespace / the trailing newline.
    return nl.strip()


def preprocess_sql(sql):
    # SQL target: strip whitespace and replace the untokenizable "<" ("<=" becomes "LESS=").
    return sql.strip().replace("<", LESS_THAN_PLACEHOLDER)


def postprocess_sql(text):
    # Inverse of preprocess_sql, applied to generated text before executing it.
    return text.replace(LESS_THAN_PLACEHOLDER, "<").strip()


def decode_sql(tokenizer, token_ids):
    # Turn generated token ids back into an executable SQL string.
    # clean_up_tokenization_spaces=False is required: the default cleanup rewrites " , " to ", ",
    # which leaves the records unchanged but makes almost every query fail SQL exact match.
    text = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return postprocess_sql(text)
