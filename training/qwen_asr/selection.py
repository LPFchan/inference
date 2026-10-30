"""Fixed development scoring and paired bootstrap gates for the single-worker loop."""
import argparse
import json
from pathlib import Path
import random

import re

import jiwer
from common import normalize

LANGUAGES=('en','ko','ja','zh')
CJK=('ko','ja','zh')


def read_evaluation(path,glosses=None):
    rows=json.loads(Path(path).read_text())['rows']
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate evaluation ids')
    rows={r['id']:r for r in rows}
    if glosses is not None:
        if set(glosses)!=set(rows):raise ValueError('gloss file covers a different evaluation set')
        for i,r in rows.items():
            if glosses[i]['reference']!=r['reference']:raise ValueError('gloss file reference differs')
            r['glosses']=glosses[i]['glosses']
    return rows


def gloss_spans(reference,glosses):
    """(start, end) of each normalized gloss in a normalized reference, left to right, not inside a Latin word."""
    spans,start=[],0
    for gloss in glosses:
        found=re.compile(r'(?<![a-z0-9])'+re.escape(gloss)+r'(?![a-z0-9])').search(reference,start) if gloss else None
        if not found:raise ValueError(f'gloss {gloss!r} is not in the reference {reference!r}')
        spans.append(found.span());start=found.end()
    return spans


def error_counts(row):
    ref,hyp=normalize(row['reference']),normalize(row['hypothesis'])

    def counts(reference):
        reference=' '.join(reference.split())
        out=jiwer.process_words(reference,hyp) if row['language'] in ('en','ko') else jiwer.process_characters(reference.replace(' ',''),hyp.replace(' ',''))
        return out.substitutions+out.deletions+out.insertions,out.hits+out.substitutions+out.deletions
    glosses=row.get('glosses') or []
    if not glosses:return counts(ref)
    # Translator glosses are optional reference spans, as in NIST sclite: the fewest errors over
    # keeping or dropping each one, over the length of the reference without any of them.
    variants,last=[''],0
    for start,end in gloss_spans(ref,glosses):
        variants=[v+ref[last:start]+kept for v in variants for kept in (ref[start:end],'')];last=end
    variants=[v+ref[last:] for v in variants]
    return min(counts(v)[0] for v in variants),counts(variants[-1])[1]


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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate',type=Path);p.add_argument('incumbent',type=Path);p.add_argument('anchor',type=Path)
    p.add_argument('--glosses',type=Path,help='frozen glosses.py file: score translator glosses as optional reference spans');a=p.parse_args()
    glosses=json.loads(a.glosses.read_text())['rows'] if a.glosses else None
    print(json.dumps(assess(*[read_evaluation(v,glosses) for v in (a.candidate,a.incumbent,a.anchor)]),indent=2))


if __name__=='__main__':main()
