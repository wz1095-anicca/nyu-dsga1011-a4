import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
import os
import re
import zlib
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


### Custom transformation: targeted QWERTY keyboard typos
# Simulates a user typing carelessly (e.g. on a phone): a word gets one typo, where one letter is
# swapped for an adjacent keyboard key. Sentiment words (SENTIMENT_WORDS) get a typo with
# probability SENTIMENT_TYPO_PROB, all other eligible words with the lower OTHER_TYPO_PROB, so the
# test probes whether the model still recognizes the words that carry the review's sentiment.
# A typo is only accepted if it does not form another real English word (e.g. "bad" -> "had"
# is rejected), so the misspelled word stays recognizable and the label is preserved.
# Both probabilities can be overridden with environment variables to compare settings in one job.
SENTIMENT_TYPO_PROB = float(os.environ.get("SENTIMENT_TYPO_PROB", 0.6))
OTHER_TYPO_PROB = float(os.environ.get("OTHER_TYPO_PROB", 0.3))
TRANSFORM_SEED = 0
MAX_TYPO_TRIES = 10

# Common, clearly polarized words in movie reviews, including frequent inflected forms because
# matching is exact. Being on this list only raises a word's typo probability; a typo never
# changes a word's meaning, so a word that is used non-sentimentally in context is harmless.
POSITIVE_WORDS = {
    "good", "great", "best", "better", "excellent", "amazing", "wonderful", "wonderfully",
    "brilliant", "brilliantly", "beautiful", "beautifully", "perfect", "perfectly", "superb",
    "superbly", "fantastic", "awesome", "love", "loved", "loves", "lovely", "enjoy", "enjoyed",
    "enjoyable", "fun", "funny", "hilarious", "favorite", "favourite", "masterpiece", "recommend",
    "recommended", "impressive", "outstanding", "terrific", "touching", "moving", "powerful",
    "fascinating", "entertaining", "nice", "gem", "classic", "incredible", "delightful",
}
NEGATIVE_WORDS = {
    "bad", "worst", "worse", "awful", "terrible", "terribly", "horrible", "horribly", "boring",
    "bored", "waste", "wasted", "poor", "poorly", "stupid", "dull", "lame", "pathetic",
    "ridiculous", "disappointing", "disappointed", "disappointment", "mess", "annoying", "weak",
    "crap", "crappy", "garbage", "unfunny", "pointless", "mediocre", "hate", "hated", "failed",
    "avoid", "worthless", "laughable", "painful", "dreadful", "cheap", "predictable", "lousy",
    "sucks", "tedious", "bland", "silly", "unwatchable", "forgettable",
}
SENTIMENT_WORDS = POSITIVE_WORDS | NEGATIVE_WORDS

# Words matched as tokens: letters, optionally one apostrophe part ("don't", "film's"); both the
# straight apostrophe (') and the curly one (’, common in pasted reviews) count as apostrophes.
# Everything else (punctuation, numbers, "<br />" HTML, spacing) is left exactly as in the original.
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")

# Words never modified: negations, contrast and intensity words carry the sentiment's polarity, so
# they stay intact to keep the label unambiguous; function words are skipped so that typos fall on
# content words.
PROTECTED_WORDS = {
    "not", "no", "nor", "never", "none", "nothing", "nobody", "neither", "nowhere", "without",
    "but", "however", "although", "though", "still", "even", "ever", "yet", "very", "too", "really", "quite", "only", "just",
    "this", "that", "these", "those", "there", "their", "they", "them", "then", "than",
    "what", "which", "who", "whom", "when", "where", "while", "with", "from", "into", "onto",
    "have", "has", "had", "been", "being", "were", "was", "will", "would", "could", "should",
    "does", "did", "doing", "about", "after", "before", "over", "under", "again", "also",
    "some", "such", "more", "most", "much", "many", "other", "each", "every", "your", "yours",
}

# Short function words that WordNet does not list; together with PROTECTED_WORDS and WordNet they
# define the "real English word" check that typos must not produce.
EXTRA_REAL_WORDS = {
    "the", "and", "for", "his", "her", "him", "she", "you", "are", "its", "our", "can", "may",
    "off", "out", "own", "any", "all", "how", "why", "one", "two", "yes", "per", "via", "few",
    "nor", "let", "got", "get", "use", "way", "who", "now", "new", "old", "bit", "lot",
}

# Nearest keys on a QWERTY keyboard, used to make realistic single-character typos.
QWERTY_NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg", "y": "tugh", "u": "yihj",
    "i": "uojk", "o": "ipkl", "p": "ol", "a": "qwsz", "s": "awedxz", "d": "serfcx",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}


def is_real_word(word):
    # A typo result counts as a real word if it is a known function word or WordNet has an entry
    # for it (WordNet also maps inflections such as "had" or "loved" to their base form).
    lower = word.lower()
    return lower in PROTECTED_WORDS or lower in EXTRA_REAL_WORDS or bool(wordnet.synsets(lower))


def add_typo(word, rng):
    # Replace one random letter that has known keyboard neighbors with an adjacent key.
    # Retry when the result is another real word; give up (keep the word) after MAX_TYPO_TRIES.
    positions = [i for i, ch in enumerate(word) if ch.lower() in QWERTY_NEIGHBORS]
    if not positions:
        return word
    for _ in range(MAX_TYPO_TRIES):
        i = rng.choice(positions)
        replacement = rng.choice(QWERTY_NEIGHBORS[word[i].lower()])
        if word[i].isupper():
            replacement = replacement.upper()
        candidate = word[:i] + replacement + word[i + 1:]
        if not is_real_word(candidate):
            return candidate
    return word


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]

    # Seed a per-example RNG from the text itself, so the same review always gets the same
    # transformation no matter the order, batching, or how many times map() is called.
    rng = random.Random(TRANSFORM_SEED * 1_000_003 + zlib.crc32(text.encode("utf-8")))

    def transform_word(match):
        word = match.group(0)
        lower = word.lower()

        # Sentiment words: high typo probability, with no length limit so "bad" or "fun" qualify.
        if lower in SENTIMENT_WORDS:
            typo_prob = SENTIMENT_TYPO_PROB
        # Other words: skip contractions, short words and protected words to keep the label
        # unambiguous; the rest get the low background typo probability.
        elif "'" in word or "’" in word or len(word) < 4 or lower in PROTECTED_WORDS:
            return word
        else:
            typo_prob = OTHER_TYPO_PROB

        if rng.random() < typo_prob:
            return add_typo(word, rng)
        return word

    # Rewrite only the matched words; all other characters stay byte-for-byte unchanged.
    example["text"] = WORD_RE.sub(transform_word, text)

    ##### YOUR CODE ENDS HERE ######

    return example
