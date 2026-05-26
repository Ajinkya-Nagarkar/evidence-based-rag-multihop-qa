"""Deep error analysis of pipeline predictions vs gold HotpotQA answers."""
import json
import re
import sys
import string
from collections import Counter, defaultdict

def normalize_answer(s):
    """HotpotQA-style normalization."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    prec = num_same / len(prediction_tokens)
    rec = num_same / len(ground_truth_tokens)
    f1 = 2 * prec * rec / (prec + rec)
    return f1, prec, rec

def sp_f1(pred_sp, gold_sp):
    pred_set = set(tuple(x) for x in pred_sp) if pred_sp else set()
    gold_set = set(tuple(x) for x in gold_sp) if gold_sp else set()
    tp = len(pred_set & gold_set)
    if tp == 0:
        return 0.0, 0.0, 0.0
    prec = tp / len(pred_set) if pred_set else 0.0
    rec = tp / len(gold_set) if gold_set else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec

def load_gold(gold_path, pred_ids):
    with open(gold_path) as f:
        data = json.load(f)
    gold = {}
    for item in data:
        qid = str(item['_id'])
        if qid in pred_ids:
            gold[qid] = {
                'answer': item['answer'],
                'sp': [[sf[0], sf[1]] for sf in item['supporting_facts']],
                'type': item.get('type', ''),
                'level': item.get('level', ''),
                'question': item.get('question', ''),
            }
    return gold

def main():
    pred_path = sys.argv[1]
    gold_path = sys.argv[2]

    with open(pred_path) as f:
        preds = json.load(f)
    
    gold = load_gold(gold_path, set(preds.keys()))
    
    total = len(preds)
    print(f"\n{'='*80}")
    print(f"ERROR ANALYSIS: {total} predictions")
    print(f"{'='*80}")
    
    # ========== 1. Answer Analysis ==========
    em_correct = 0
    em_wrong = []
    partial_match = []
    wrong_type = []
    abstained = []
    yesno_errors = []
    answer_too_long = []
    
    type_stats = defaultdict(lambda: {'total': 0, 'em_correct': 0, 'f1_sum': 0.0})
    level_stats = defaultdict(lambda: {'total': 0, 'em_correct': 0, 'f1_sum': 0.0})
    
    for qid, pred in preds.items():
        if qid not in gold:
            continue
        g = gold[qid]
        pred_ans = pred.get('answer', '')
        gold_ans = g['answer']
        qtype = g['type']
        level = g['level']
        question = g['question']
        
        f1, prec, rec = f1_score(pred_ans, gold_ans)
        em = 1.0 if normalize_answer(pred_ans) == normalize_answer(gold_ans) else 0.0
        
        type_stats[qtype]['total'] += 1
        type_stats[qtype]['f1_sum'] += f1
        level_stats[level]['total'] += 1
        level_stats[level]['f1_sum'] += f1
        
        if em == 1.0:
            em_correct += 1
            type_stats[qtype]['em_correct'] += 1
            level_stats[level]['em_correct'] += 1
        else:
            entry = {
                'qid': qid,
                'question': question[:100],
                'predicted': pred_ans[:80],
                'gold': gold_ans[:80],
                'f1': round(f1, 3),
                'type': qtype,
                'level': level,
            }
            em_wrong.append(entry)
            
            # Classify error type
            if pred_ans.lower().strip() in ('cannot answer based on the provided evidence.', 'cannot determine from evidence', ''):
                abstained.append(entry)
            elif gold_ans.lower() in ('yes', 'no') and pred_ans.lower().strip() not in ('yes', 'no'):
                yesno_errors.append(entry)
            elif gold_ans.lower() not in ('yes', 'no') and pred_ans.lower().strip() in ('yes', 'no'):
                wrong_type.append(entry)
            elif f1 > 0.5:
                partial_match.append(entry)
            elif len(pred_ans.split()) > 10:
                answer_too_long.append(entry)
    
    print(f"\n--- ANSWER METRICS ---")
    print(f"EM correct: {em_correct}/{total} ({em_correct/total*100:.1f}%)")
    print(f"EM wrong:   {len(em_wrong)}/{total} ({len(em_wrong)/total*100:.1f}%)")
    
    print(f"\n--- ERROR BREAKDOWN ---")
    print(f"  Abstained/empty:      {len(abstained)} ({len(abstained)/total*100:.1f}%)")
    print(f"  Yes/No gold, wrong:   {len(yesno_errors)} ({len(yesno_errors)/total*100:.1f}%)")
    print(f"  Wrong answer type:    {len(wrong_type)} ({len(wrong_type)/total*100:.1f}%)")
    print(f"  Partial match (F1>0.5): {len(partial_match)} ({len(partial_match)/total*100:.1f}%)")
    print(f"  Answer too long:      {len(answer_too_long)} ({len(answer_too_long)/total*100:.1f}%)")
    completely_wrong = len(em_wrong) - len(abstained) - len(yesno_errors) - len(wrong_type) - len(partial_match) - len(answer_too_long)
    print(f"  Completely wrong:     {completely_wrong} ({completely_wrong/total*100:.1f}%)")
    
    print(f"\n--- BY QUESTION TYPE ---")
    for qtype, stats in sorted(type_stats.items()):
        n = stats['total']
        em_rate = stats['em_correct'] / n * 100 if n else 0
        avg_f1 = stats['f1_sum'] / n if n else 0
        print(f"  {qtype:>12s}: n={n:>3d}  EM={em_rate:>5.1f}%  avgF1={avg_f1:.3f}")
    
    print(f"\n--- BY DIFFICULTY ---")
    for level, stats in sorted(level_stats.items()):
        n = stats['total']
        em_rate = stats['em_correct'] / n * 100 if n else 0
        avg_f1 = stats['f1_sum'] / n if n else 0
        print(f"  {level:>8s}: n={n:>3d}  EM={em_rate:>5.1f}%  avgF1={avg_f1:.3f}")
    
    # ========== 2. Supporting Facts Analysis ==========
    print(f"\n--- SUPPORTING FACTS ANALYSIS ---")
    sp_em_correct = 0
    sp_empty = 0
    sp_too_few = 0
    sp_too_many = 0
    sp_right_title_wrong_idx = 0
    sp_wrong_title = 0
    
    sp_counts_pred = []
    sp_counts_gold = []
    
    for qid, pred in preds.items():
        if qid not in gold:
            continue
        g = gold[qid]
        pred_sp = pred.get('sp', [])
        gold_sp = g['sp']
        
        sp_counts_pred.append(len(pred_sp))
        sp_counts_gold.append(len(gold_sp))
        
        pred_sp_set = set(tuple(x) for x in pred_sp) if pred_sp else set()
        gold_sp_set = set(tuple(x) for x in gold_sp) if gold_sp else set()
        
        if pred_sp_set == gold_sp_set:
            sp_em_correct += 1
        
        if not pred_sp:
            sp_empty += 1
            continue
            
        if len(pred_sp) < len(gold_sp):
            sp_too_few += 1
        elif len(pred_sp) > len(gold_sp):
            sp_too_many += 1
        
        # Check title vs index errors
        pred_titles = set(x[0] for x in pred_sp)
        gold_titles = set(x[0] for x in gold_sp)
        
        if pred_titles & gold_titles and pred_sp_set != gold_sp_set:
            # Some titles match but not exact SP match
            for gsp in gold_sp:
                if gsp[0] in pred_titles and tuple(gsp) not in pred_sp_set:
                    sp_right_title_wrong_idx += 1
        
        missing_titles = gold_titles - pred_titles
        if missing_titles:
            sp_wrong_title += 1
    
    avg_pred_sp = sum(sp_counts_pred) / len(sp_counts_pred) if sp_counts_pred else 0
    avg_gold_sp = sum(sp_counts_gold) / len(sp_counts_gold) if sp_counts_gold else 0
    
    print(f"  SP EM correct:       {sp_em_correct}/{total} ({sp_em_correct/total*100:.1f}%)")
    print(f"  SP empty:            {sp_empty}/{total} ({sp_empty/total*100:.1f}%)")
    print(f"  SP too few:          {sp_too_few}/{total} ({sp_too_few/total*100:.1f}%)")
    print(f"  SP too many:         {sp_too_many}/{total} ({sp_too_many/total*100:.1f}%)")
    print(f"  Right title/wrong idx: {sp_right_title_wrong_idx} facts")
    print(f"  Missing title(s):   {sp_wrong_title}/{total} ({sp_wrong_title/total*100:.1f}%)")
    print(f"  Avg pred SP count:   {avg_pred_sp:.1f} (gold avg: {avg_gold_sp:.1f})")
    
    # ========== 3. Detailed error examples ==========
    print(f"\n{'='*80}")
    print(f"SAMPLE WRONG ANSWERS (worst F1 first)")
    print(f"{'='*80}")
    em_wrong.sort(key=lambda x: x['f1'])
    for entry in em_wrong[:15]:
        qid = entry['qid']
        g = gold[qid]
        p = preds[qid]
        print(f"\n[{entry['type']}|{entry['level']}] F1={entry['f1']}")
        print(f"  Q: {entry['question']}")
        print(f"  Gold: '{entry['gold']}'")
        print(f"  Pred: '{entry['predicted']}'")
        gold_titles = list(set(sf[0] for sf in g['sp']))
        pred_titles = list(set(sf[0] for sf in p.get('sp', [])))
        print(f"  Gold SP titles: {gold_titles}")
        print(f"  Pred SP titles: {pred_titles}")
    
    # ========== 4. Yes/No specific analysis ==========
    print(f"\n{'='*80}")
    print(f"YES/NO QUESTION ANALYSIS")
    print(f"{'='*80}")
    yn_total = sum(1 for qid in preds if qid in gold and gold[qid]['answer'].lower() in ('yes', 'no'))
    yn_correct = 0
    yn_wrong_ans = 0
    yn_non_yn_pred = 0
    for qid, pred in preds.items():
        if qid not in gold:
            continue
        g = gold[qid]
        if g['answer'].lower() in ('yes', 'no'):
            pred_ans = pred.get('answer', '').lower().strip()
            if normalize_answer(pred_ans) == normalize_answer(g['answer']):
                yn_correct += 1
            elif pred_ans in ('yes', 'no'):
                yn_wrong_ans += 1
            else:
                yn_non_yn_pred += 1
    print(f"  Total yes/no questions: {yn_total}")
    print(f"  Correct:   {yn_correct} ({yn_correct/max(yn_total,1)*100:.1f}%)")
    print(f"  Wrong y/n: {yn_wrong_ans} ({yn_wrong_ans/max(yn_total,1)*100:.1f}%)")
    print(f"  Non-y/n pred: {yn_non_yn_pred} ({yn_non_yn_pred/max(yn_total,1)*100:.1f}%)")
    
    # ========== 5. Entity answer analysis ==========
    print(f"\n{'='*80}")
    print(f"ENTITY ANSWER ANALYSIS")
    print(f"{'='*80}")
    entity_total = sum(1 for qid in preds if qid in gold and gold[qid]['answer'].lower() not in ('yes', 'no'))
    entity_em = 0
    entity_partial = 0
    entity_wrong = 0
    for qid, pred in preds.items():
        if qid not in gold:
            continue
        g = gold[qid]
        if g['answer'].lower() not in ('yes', 'no'):
            pred_ans = pred.get('answer', '')
            f1, _, _ = f1_score(pred_ans, g['answer'])
            if normalize_answer(pred_ans) == normalize_answer(g['answer']):
                entity_em += 1
            elif f1 > 0.5:
                entity_partial += 1
            else:
                entity_wrong += 1
    print(f"  Total entity questions: {entity_total}")
    print(f"  EM correct:    {entity_em} ({entity_em/max(entity_total,1)*100:.1f}%)")
    print(f"  Partial (F1>0.5): {entity_partial} ({entity_partial/max(entity_total,1)*100:.1f}%)")
    print(f"  Wrong:         {entity_wrong} ({entity_wrong/max(entity_total,1)*100:.1f}%)")
    
    # ========== 6. Verification analysis ==========
    # Only if predictions contain verification results (from the new pipeline)
    has_verification = any(
        pred.get('verification') is not None 
        for pred in preds.values()
    )
    
    if has_verification:
        print(f"\n{'='*80}")
        print(f"VERIFICATION ANALYSIS")
        print(f"{'='*80}")
        
        v_supported = 0
        v_unsupported = 0
        v_missing = 0
        v_scores = []
        
        # Track EM by verification status
        supported_em = 0
        supported_total = 0
        unsupported_em = 0
        unsupported_total = 0
        
        # Track verification by question type
        v_type_stats = defaultdict(lambda: {'supported': 0, 'unsupported': 0, 'scores': []})
        
        for qid, pred in preds.items():
            if qid not in gold:
                continue
            g = gold[qid]
            vr = pred.get('verification')
            pred_ans = pred.get('answer', '')
            em = 1.0 if normalize_answer(pred_ans) == normalize_answer(g['answer']) else 0.0
            qtype = g['type']
            
            if vr is None:
                v_missing += 1
                continue
            
            score = vr.get('support_score', 0.0)
            is_supported = vr.get('is_supported', False)
            v_scores.append(score)
            
            if is_supported:
                v_supported += 1
                supported_total += 1
                supported_em += em
                v_type_stats[qtype]['supported'] += 1
            else:
                v_unsupported += 1
                unsupported_total += 1
                unsupported_em += em
                v_type_stats[qtype]['unsupported'] += 1
            
            v_type_stats[qtype]['scores'].append(score)
        
        v_total = v_supported + v_unsupported
        if v_total > 0:
            avg_score = sum(v_scores) / len(v_scores)
            print(f"\n--- OVERALL VERIFICATION ---")
            print(f"  Verified predictions:  {v_total}")
            print(f"  Supported:   {v_supported}/{v_total} ({v_supported/v_total*100:.1f}%)")
            print(f"  Unsupported: {v_unsupported}/{v_total} ({v_unsupported/v_total*100:.1f}%)")
            if v_missing:
                print(f"  Missing:     {v_missing}")
            print(f"  Avg support score:   {avg_score:.4f}")
            print(f"  Min support score:   {min(v_scores):.4f}")
            print(f"  Max support score:   {max(v_scores):.4f}")
            
            print(f"\n--- VERIFICATION vs CORRECTNESS ---")
            if supported_total > 0:
                print(f"  Supported answers:   EM={supported_em/supported_total*100:.1f}% (n={supported_total})")
            if unsupported_total > 0:
                print(f"  Unsupported answers: EM={unsupported_em/unsupported_total*100:.1f}% (n={unsupported_total})")
            
            # This is the key diagnostic: if supported EM >> unsupported EM,
            # the verifier is discriminating well and retries help
            if supported_total > 0 and unsupported_total > 0:
                gap = (supported_em/supported_total - unsupported_em/unsupported_total) * 100
                if gap > 5:
                    print(f"  → Verifier discriminates well (+{gap:.1f}pp EM gap)")
                elif gap > 0:
                    print(f"  → Verifier has weak discrimination (+{gap:.1f}pp EM gap)")
                else:
                    print(f"  → Verifier not discriminating ({gap:.1f}pp EM gap)")
            
            print(f"\n--- VERIFICATION BY QUESTION TYPE ---")
            for qtype, stats in sorted(v_type_stats.items()):
                n = stats['supported'] + stats['unsupported']
                avg_s = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
                sup_rate = stats['supported'] / n * 100 if n else 0
                print(f"  {qtype:>12s}: n={n:>3d}  supported={sup_rate:>5.1f}%  avg_score={avg_s:.3f}")
            
            # Show worst unsupported but correct answers (verifier false negatives)
            false_negatives = []
            for qid, pred in preds.items():
                if qid not in gold:
                    continue
                g = gold[qid]
                vr = pred.get('verification')
                if vr is None:
                    continue
                pred_ans = pred.get('answer', '')
                em = normalize_answer(pred_ans) == normalize_answer(g['answer'])
                if em and not vr.get('is_supported', True):
                    false_negatives.append({
                        'qid': qid,
                        'answer': pred_ans[:60],
                        'score': vr.get('support_score', 0),
                        'type': g['type'],
                    })
            
            if false_negatives:
                false_negatives.sort(key=lambda x: x['score'])
                print(f"\n--- VERIFIER FALSE NEGATIVES (correct but marked unsupported) ---")
                print(f"  Total: {len(false_negatives)}")
                for entry in false_negatives[:5]:
                    print(f"  [{entry['type']}] score={entry['score']:.3f} answer='{entry['answer']}'")

if __name__ == '__main__':
    main()
