"""
Custom evaluation script for hallucination-resistant multi-hop QA pipeline.
Computes answer metrics (EM, F1, Fuzzy, Containment, BERTScore) and
support metrics (SP EM, SP F1, Title EM, Title F1, Title Containment, Title Fuzzy).

Usage:
    python scripts/evaluate_custom.py \
        --predictions results/predictions_v39_final.json \
        --gold data/hotpot_dev_distractor_v1.json \
        --output results/metrics_v39_custom.json
"""

import json
import re
import string
import argparse
from pathlib import Path
from rapidfuzz import fuzz
from bert_score import score as bert_score

FAIL = {'Error occurred during processing.', 'Cannot answer based on the provided evidence.', ''}


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def token_f1(pred: str, gold: str) -> float:
    pt = normalize(pred).split()
    gt = normalize(gold).split()
    common = set(pt) & set(gt)
    if not common:
        return 0.0
    p = len(common) / len(pt)
    r = len(common) / len(gt)
    return 2 * p * r / (p + r)


def evaluate(predictions_path: str, gold_path: str, output_path: str = None,
             fuzzy_threshold: int = 75, bert: bool = True):

    with open(predictions_path) as f:
        preds = json.load(f)
    with open(gold_path) as f:
        gold_data = json.load(f)
    gold_map = {ex['_id']: ex for ex in gold_data}

    total = len(preds)
    pairs, failed_ids = [], []

    for qid, p in preds.items():
        ans = p.get('answer', '')
        gold_ans = p.get('gold_answer', '') or (gold_map[qid]['answer'] if qid in gold_map else '')
        if ans in FAIL or not ans:
            failed_ids.append(qid)
        else:
            pairs.append((qid, ans, gold_ans))

    valid = len(pairs)
    failed = len(failed_ids)

    # ── Answer metrics ──
    em = f1s = fuz = cont = close = wrong = 0
    for qid, ans, gold in pairs:
        n_ans, n_gold = normalize(ans), normalize(gold)
        if n_ans == n_gold:
            em += 1
        else:
            r = fuzz.token_set_ratio(ans.lower(), gold.lower())
            if r >= fuzzy_threshold:
                close += 1
            else:
                wrong += 1
        f1s += token_f1(ans, gold)
        fuz += fuzz.token_set_ratio(ans.lower(), gold.lower()) / 100.0
        if n_ans and n_gold and (n_ans in n_gold or n_gold in n_ans):
            cont += 1

    errors = valid - em
    ans_list = [p[1] for p in pairs]
    gold_list = [p[2] for p in pairs]

    # ── BERTScore ──
    bert_p = bert_r = bert_f = b90 = b85 = b80 = 0.0
    if bert:
        print(f"Computing BERTScore on {valid} pairs...")
        P, R, F = bert_score(ans_list, gold_list, lang='en', verbose=False)
        bert_p = P.mean().item()
        bert_r = R.mean().item()
        bert_f = F.mean().item()
        fscores = F.tolist()
        b90 = sum(1 for s in fscores if s >= 0.90) / total
        b85 = sum(1 for s in fscores if s >= 0.85) / total
        b80 = sum(1 for s in fscores if s >= 0.80) / total

    # ── Support metrics ──
    sp_em = sp_f1s = sp_cont = t_em = t_f1s = t_cont = t_fuz = sp_valid = 0
    for qid, p in preds.items():
        if p.get('answer', '') in FAIL or qid not in gold_map:
            continue
        pred_sp = p.get('sp', [])
        gold_sp = gold_map[qid].get('supporting_facts', [])
        pred_set   = set(tuple(s) for s in pred_sp if isinstance(s, list) and len(s) == 2)
        gold_set   = set(tuple(s) for s in gold_sp if isinstance(s, list) and len(s) == 2)
        pred_titles = set(s[0] for s in pred_sp if isinstance(s, list) and len(s) >= 1)
        gold_titles = set(s[0] for s in gold_sp if isinstance(s, list) and len(s) >= 1)
        sp_valid += 1

        if pred_set == gold_set:
            sp_em += 1
        if pred_set and gold_set:
            c = pred_set & gold_set
            pr = len(c) / len(pred_set); rc = len(c) / len(gold_set)
            sp_f1s += 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0
        if pred_set and gold_set and (pred_set <= gold_set or gold_set <= pred_set):
            sp_cont += 1
        if pred_titles == gold_titles:
            t_em += 1
        if pred_titles and gold_titles:
            c = pred_titles & gold_titles
            pr = len(c) / len(pred_titles); rc = len(c) / len(gold_titles)
            t_f1s += 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0
        if pred_titles and gold_titles and (pred_titles <= gold_titles or gold_titles <= pred_titles):
            t_cont += 1
        if gold_titles and pred_titles:
            scores_t = [max(fuzz.token_set_ratio(gt.lower(), pt.lower()) for pt in pred_titles)
                        for gt in gold_titles]
            t_fuz += sum(scores_t) / len(scores_t) / 100.0

    results = {
        'summary': {'total': total, 'valid': valid, 'failed': failed},
        'answer': {
            'em_valid':          round(em / valid, 4),
            'f1_valid':          round(f1s / valid, 4),
            'fuzzy_valid':       round(fuz / valid, 4),
            'containment_valid': round(cont / valid, 4),
            'bertscore_p':       round(bert_p, 4),
            'bertscore_r':       round(bert_r, 4),
            'bertscore_f':       round(bert_f, 4),
            'bertscore_ge90':    round(b90, 4),
            'bertscore_ge85':    round(b85, 4),
            'bertscore_ge80':    round(b80, 4),
            f'errors_close_fuzzy{fuzzy_threshold}': close,
            f'errors_wrong_fuzzy{fuzzy_threshold}': wrong,
        },
        'support': {
            'sp_em_set':         round(sp_em / sp_valid, 4),
            'sp_f1_set':         round(sp_f1s / sp_valid, 4),
            'sp_containment':    round(sp_cont / sp_valid, 4),
            'title_em':          round(t_em / sp_valid, 4),
            'title_f1':          round(t_f1s / sp_valid, 4),
            'title_containment': round(t_cont / sp_valid, 4),
            'title_fuzzy':       round(t_fuz / sp_valid, 4),
        }
    }

    # ── Print ──
    print(f"\n=== CUSTOM METRICS (n={total}) ===")
    print(f"\n--- Answer (valid={valid}, failed={failed}) ---")
    for k, v in results['answer'].items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v*100:.1f}%")
        else:
            print(f"  {k:30s}: {v}")
    print(f"\n--- Support (n={sp_valid}) ---")
    for k, v in results['support'].items():
        print(f"  {k:30s}: {v*100:.1f}%")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--gold', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--fuzzy-threshold', type=int, default=75)
    parser.add_argument('--no-bert', action='store_true')
    args = parser.parse_args()

    evaluate(
        predictions_path=args.predictions,
        gold_path=args.gold,
        output_path=args.output,
        fuzzy_threshold=args.fuzzy_threshold,
        bert=not args.no_bert,
    )
