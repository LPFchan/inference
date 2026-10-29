"""Fixed development scoring and paired bootstrap gates for the single-worker loop."""
import argparse
import json
from pathlib import Path
import random

import jiwer
from common import normalize

LANGUAGES=('en','ko','ja','zh')
CJK=('ko','ja','zh')


def read_evaluation(path):
    rows=json.loads(Path(path).read_text())['rows']
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate evaluation ids')
    return {r['id']:r for r in rows}


def error_counts(row):
    ref,hyp=normalize(row['reference']),normalize(row['hypothesis'])
    out=jiwer.process_words(ref,hyp) if row['language'] in ('en','ko') else jiwer.process_characters(ref.replace(' ',''),hyp.replace(' ',''))
    return out.substitutions+out.deletions+out.insertions,out.hits+out.substitutions+out.deletions


def rates(rows):
    result={}
    for lang in LANGUAGES:
        pairs=[error_counts(r) for r in rows.values() if r['language']==lang]
        denominator=sum(n for _,n in pairs)
        if not denominator:raise ValueError(f'missing reference denominator: {lang}')
        result[lang]=sum(e for e,_ in pairs)/denominator
    return result


def aligned(*collections):
    ids=set(collections[0])
    for rows in collections[1:]:
        if set(rows)!=ids:raise ValueError('evaluation sample sets differ')
        if any((r['reference'],r['language'])!=(collections[0][i]['reference'],collections[0][i]['language']) for i,r in rows.items()):
            raise ValueError('evaluation references differ')


def normalized_score(errors,anchor):
    if any(anchor[l]<=0 for l in CJK):raise ValueError('zero anchor error cannot be ratio-normalized')
    return sum(errors[l]/anchor[l] for l in CJK)/3


def assess(candidate,incumbent,anchor,repetitions=400,seed=20261029):
    aligned(candidate,incumbent,anchor)
    cr,ir,ar=rates(candidate),rates(incumbent),rates(anchor)
    cs,ins=normalized_score(cr,ar),normalized_score(ir,ar)
    gates={
        'one_percent_relative_gain':cs<ins*0.99,
        'cjk_regression_limits':all(cr[l]<=min(ir[l],ar[l])*1.02 for l in CJK),
        'english_limit':cr['en']<=min(ir['en'],ar['en'])+0.005,
        'empty_output_limit':sum(not r['hypothesis'].strip() for r in candidate.values())<=sum(not r['hypothesis'].strip() for r in incumbent.values()),
        'truncation_limit':sum(not r.get('ended',True) for r in candidate.values())<=sum(not r.get('ended',True) for r in incumbent.values())}
    rng=random.Random(seed);deltas=[]
    counts={l:[(error_counts(candidate[i]),error_counts(incumbent[i])) for i in sorted(candidate) if candidate[i]['language']==l] for l in CJK}
    for _ in range(repetitions):
        delta=0
        for l,pairs in counts.items():
            chosen=rng.choices(pairs,k=len(pairs));ce=sum(p[0][0] for p in chosen);ie=sum(p[1][0] for p in chosen);den=sum(p[0][1] for p in chosen)
            delta+=(ce-ie)/den/ar[l]/3
        deltas.append(delta)
    deltas.sort();lower=deltas[int(0.025*len(deltas))];upper=deltas[min(len(deltas)-1,int(0.975*len(deltas)))];gates['paired_ci_below_zero']=upper<0
    return {'passes':all(gates.values()),'gates':gates,'candidate_score':cs,'incumbent_score':ins,
            'candidate_errors':cr,'incumbent_errors':ir,'anchor_errors':ar,'paired_delta_ci95':[lower,upper],
            'clips':{l:sum(r['language']==l for r in candidate.values()) for l in LANGUAGES}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate',type=Path);p.add_argument('incumbent',type=Path);p.add_argument('anchor',type=Path);a=p.parse_args()
    print(json.dumps(assess(*[read_evaluation(v) for v in (a.candidate,a.incumbent,a.anchor)]),indent=2))


if __name__=='__main__':main()
