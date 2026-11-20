'''
Diagnose the SQL queries that hit the evaluation timeout in utils.compute_records.

For every dev query whose saved error message is "Query timed out", re-run the generated query and
the gold query on their own and time them, so we can tell "correct but slow" queries (which a larger
timeout would rescue) from runaway queries (e.g. a missing join condition) that never finish.

Usage (from part2/):  python diagnose_timeouts.py --experiment_name baseline [--cap_secs 300]
'''
import argparse
import pickle
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

from utils import DB_PATH, read_queries


def timed_execute(query, cap_secs):
    # Run one query on a fresh connection and return (status, seconds, records).
    # A progress handler aborts the query after cap_secs, so a runaway query cannot hang this script.
    conn = sqlite3.connect(DB_PATH)
    start = time.time()
    conn.set_progress_handler(lambda: int(time.time() - start > cap_secs), 100_000)
    try:
        rows = conn.execute(query).fetchall()
        status = "ok"
    except sqlite3.OperationalError as e:
        rows = None
        status = f"capped (>{cap_secs}s)" if "interrupted" in str(e) else f"error: {e}"
    except Exception as e:
        rows = None
        status = f"error: {type(e).__name__}: {e}"
    finally:
        conn.close()
    return status, time.time() - start, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="baseline")
    parser.add_argument("--model_type", default="ft", choices=["ft", "scr"])
    parser.add_argument("--cap_secs", type=float, default=300)
    args = parser.parse_args()

    # Load the generated dev queries and their saved records/error messages from the last evaluation.
    prefix = f"t5_{args.model_type}_{args.experiment_name}_dev"
    pred_qs = read_queries(f"results/{prefix}.sql")
    with open(f"records/{prefix}.pkl", "rb") as f:
        _, pred_errors = pickle.load(f)
    gold_qs = read_queries("data/dev.sql")
    with open("records/ground_truth_dev.pkl", "rb") as f:
        gold_records, _ = pickle.load(f)

    # Keep only the queries that the evaluation recorded as timed out.
    timed_out = [i for i, msg in enumerate(pred_errors) if msg == "Query timed out"]
    print(f"{len(timed_out)}/{len(pred_qs)} generated dev queries were recorded as 'Query timed out'")
    if not timed_out:
        return

    # Re-run the generated and the gold version of each timed-out query in parallel (each on its own
    # connection), so the whole diagnosis finishes in about cap_secs. Parallel runs share the CPU, so the
    # times are upper bounds of what each query needs alone.
    jobs = [(i, kind, q) for i in timed_out for kind, q in (("pred", pred_qs[i]), ("gold", gold_qs[i]))]
    with ThreadPoolExecutor(len(jobs)) as pool:
        results = list(pool.map(lambda job: timed_execute(job[2], args.cap_secs), jobs))
    by_key = {(i, kind): res for (i, kind, _), res in zip(jobs, results)}

    # Report each query: how long the prediction and the gold query take, and whether the prediction's
    # records equal the gold records (i.e. it would have been counted correct with a larger timeout).
    for i in timed_out:
        p_status, p_secs, p_rows = by_key[(i, "pred")]
        g_status, g_secs, _ = by_key[(i, "gold")]
        if p_rows is None:
            verdict = "unknown (prediction did not finish)"
        else:
            verdict = "CORRECT" if set(p_rows) == set(gold_records[i]) else "wrong"
        print(f"\n--- dev #{i} ---")
        print(f"pred: {p_status}, {p_secs:.1f}s, rows={None if p_rows is None else len(p_rows)}, records vs gold: {verdict}")
        print(f"gold: {g_status}, {g_secs:.1f}s, rows={len(gold_records[i])}")
        print(f"PRED SQL: {pred_qs[i][:300]}")
        print(f"GOLD SQL: {gold_qs[i][:300]}")


if __name__ == "__main__":
    main()
