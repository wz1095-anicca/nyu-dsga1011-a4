'''
Q6 error analysis: compare the model's dev predictions with the gold SQL and count error types.

Every dev query whose records differ from the gold records is "wrong". Each wrong query gets exactly
one error type: the first matching check in classify(), ordered from the most basic failure (the
query cannot run) to the most local one (only a constant differs), so the counts add up to the
number of wrong queries. Statistics are COUNT/TOTAL with TOTAL = number of dev queries.

Usage (from part2/):
  python error_analysis.py --pred_sql results/t5_ft_baseline_greedy_dev.sql \
                           --pred_pkl records/t5_ft_baseline_greedy_dev.pkl
'''
import argparse
import pickle
import re
from collections import Counter

from utils import read_queries

# A table occurrence looks like "flight flight_1": the table name followed by its numbered alias.
TABLE_RE = re.compile(r"\b(\w+) \1_\d+\b")
# Quoted literals such as 'BOSTON' or 'CO' (cities, airlines, day names, ...).
STRING_RE = re.compile(r"'[^']*'")
# Bare numbers such as times (1200), flight numbers or years; aliases like flight_1 do not match
# because "_" is a word character, so there is no word boundary before the digit.
NUMBER_RE = re.compile(r"\b\d+\b")


def tables(sql):
    return Counter(TABLE_RE.findall(sql))


def select_clause(sql):
    # Text between the first SELECT and the first FROM: what the query returns.
    match = re.match(r"\s*SELECT\s+(.*?)\s+FROM\s", sql, flags=re.S)
    return match.group(1) if match else sql


# Gold query used by the dataset for questions without a real annotation; it always returns no rows.
PLACEHOLDER_GOLD = "SELECT state_1.state_code FROM state state_1 WHERE state_1.state_code = 'ASD';"


def classify(pred, gold, error_msg):
    # Return the single error type of one wrong prediction: the first check that matches.
    if gold.strip() == PLACEHOLDER_GOLD:
        # The question has no real gold SQL, so comparing tables/literals would be meaningless.
        return "Placeholder gold query (unanswerable question)"
    if error_msg == "Query timed out":
        return "Runaway query (timed out)"
    if error_msg:
        return "SQL execution error"
    if tables(pred) != tables(gold):
        return "Missing or extra table"
    if select_clause(pred) != select_clause(gold):
        return "Wrong SELECT target"
    if STRING_RE.findall(pred) != STRING_RE.findall(gold):
        return "Wrong string literal"
    if Counter(NUMBER_RE.findall(pred)) != Counter(NUMBER_RE.findall(gold)):
        return "Wrong numeric/time constant"
    return "Other (same tables/literals, different conditions)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_sql", default="results/t5_ft_baseline_greedy_dev.sql")
    parser.add_argument("--pred_pkl", default="records/t5_ft_baseline_greedy_dev.pkl")
    parser.add_argument("--examples", type=int, default=3, help="examples printed per error type")
    args = parser.parse_args()

    # Load predictions, their saved records/error messages, the gold SQL and the gold records.
    pred_qs = read_queries(args.pred_sql)
    with open(args.pred_pkl, "rb") as f:
        pred_recs, pred_errs = pickle.load(f)
    gold_qs = read_queries("data/dev.sql")
    nl_qs = read_queries("data/dev.nl")
    with open("records/ground_truth_dev.pkl", "rb") as f:
        gold_recs, _ = pickle.load(f)
    total = len(gold_qs)
    assert len(pred_qs) == total, f"{len(pred_qs)} predictions vs {total} gold queries"

    # A prediction is wrong when its records differ from the gold records (the Record EM criterion).
    wrong = [i for i in range(total) if set(pred_recs[i]) != set(gold_recs[i])]
    by_type = {}
    for i in wrong:
        by_type.setdefault(classify(pred_qs[i], gold_qs[i], pred_errs[i]), []).append(i)

    # Breakdown of execution errors by the first part of the SQLite message, e.g. "no such column".
    exec_errors = Counter(re.sub(r":\s.*", "", pred_errs[i].split(": ", 1)[-1])
                          for i in by_type.get("SQL execution error", []))

    print(f"Wrong records: {len(wrong)}/{total}")
    for t, ids in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        print(f"{t:55s} {len(ids):4d}/{total}  ({len(ids) / len(wrong):.1%} of wrong)")
    print(f"\nSQL execution error messages: {dict(exec_errors)}")

    # Print the shortest examples of each type, so they are easy to quote in Table 5.
    for t, ids in by_type.items():
        print(f"\n===== {t} =====")
        for i in sorted(ids, key=lambda i: len(gold_qs[i]))[:args.examples]:
            print(f"--- dev #{i}: {nl_qs[i]}")
            if pred_errs[i]:
                print(f"ERROR: {pred_errs[i]}")
            print(f"PRED: {pred_qs[i]}")
            print(f"GOLD: {gold_qs[i]}")


if __name__ == "__main__":
    main()
