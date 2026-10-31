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


def respace(reference,hypothesis):
    """The hypothesis with Korean spacing made optional, as in KsponSpeech's space-normalized WER:
    characters aligned to the reference break words where it does, and inserted ones where the hypothesis does."""
    def starts(text):
        out,i=set(),0
        for word in text.split():out.add(i);i+=len(word)
        return out
    chars=hypothesis.replace(' ','')
    if not chars:return ''
    ref_starts,hyp_starts,words=starts(reference),starts(hypothesis),''
    for c in jiwer.process_characters(reference.replace(' ',''),chars).alignments[0]:
        for k in range(c.hyp_end_idx-c.hyp_start_idx):
            h=c.hyp_start_idx+k
            new=h in hyp_starts if c.type=='insert' else c.ref_start_idx+k in ref_starts
            words+=(' ' if new and words else '')+chars[h]
    return words


def error_counts(row):
    ref,hyp=normalize(row['reference']),normalize(row['hypothesis'])

    def counts(reference):
        reference=' '.join(reference.split())
        if row['language']=='en':out=jiwer.process_words(reference,hyp)
        elif row['language']=='ko':out=jiwer.process_words(reference,respace(reference,hyp))
        else:out=jiwer.process_characters(reference.replace(' ',''),hyp.replace(' ',''))
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


def rates(*runs):
    result={}
    for lang in LANGUAGES:
        pairs=[error_counts(r) for rows in runs for r in rows.values() if r['language']==lang]
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


def pooled(runs,i):
    return tuple(map(sum,zip(*(error_counts(rows[i]) for rows in runs))))


def assess(candidate,incumbent,anchor,repetitions=400,seed=20261029):
    """Gate candidate against incumbent. Each side is one evaluation, or the same number of
    independent seeds of one recipe, whose errors pool per clip before every gate."""
    cands=[candidate] if isinstance(candidate,dict) else list(candidate)
    incs=[incumbent] if isinstance(incumbent,dict) else list(incumbent)
    if len(cands)!=len(incs):raise ValueError('candidate and incumbent need the same number of seeds')
    aligned(*cands,*incs,anchor)
    cr,ir,ar=rates(*cands),rates(*incs),rates(anchor)
    cs,ins=normalized_score(cr,ar),normalized_score(ir,ar)
    gates={
        'one_percent_relative_gain':cs<ins*0.99,
        'cjk_regression_limits':all(cr[l]<=min(ir[l],ar[l])*1.02 for l in CJK),
        'english_limit':cr['en']<=min(ir['en'],ar['en'])+0.005,
        'empty_output_limit':sum(not r['hypothesis'].strip() for c in cands for r in c.values())<=sum(not r['hypothesis'].strip() for c in incs for r in c.values()),
        'truncation_limit':sum(not r.get('ended',True) for c in cands for r in c.values())<=sum(not r.get('ended',True) for c in incs for r in c.values())}
    rng=random.Random(seed);deltas=[]
    counts={l:[(pooled(cands,i),pooled(incs,i)) for i in sorted(anchor) if anchor[i]['language']==l] for l in CJK}
    for _ in range(repetitions):
        delta=0
        for l,pairs in counts.items():
            chosen=rng.choices(pairs,k=len(pairs));ce=sum(p[0][0] for p in chosen);ie=sum(p[1][0] for p in chosen);den=sum(p[0][1] for p in chosen)
            delta+=(ce-ie)/den/ar[l]/3
        deltas.append(delta)
    deltas.sort();lower=deltas[int(0.025*len(deltas))];upper=deltas[min(len(deltas)-1,int(0.975*len(deltas)))];gates['paired_ci_below_zero']=upper<0
    return {'passes':all(gates.values()),'gates':gates,'candidate_score':cs,'incumbent_score':ins,
            'candidate_errors':cr,'incumbent_errors':ir,'anchor_errors':ar,'paired_delta_ci95':[lower,upper],
            'clips':{l:sum(r['language']==l for r in anchor.values()) for l in LANGUAGES},'seeds':len(cands)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate',type=Path);p.add_argument('incumbent',type=Path);p.add_argument('anchor',type=Path)
    p.add_argument('--glosses',type=Path,help='frozen glosses.py file: score translator glosses as optional reference spans')
    p.add_argument('--second-seeds',type=Path,nargs=2,metavar=('CANDIDATE','INCUMBENT'),
                   help='second-seed evaluations of both recipes: pool each recipe\'s two seeds per clip');a=p.parse_args()
    glosses=json.loads(a.glosses.read_text())['rows'] if a.glosses else None
    read=lambda v:read_evaluation(v,glosses)
    candidate,incumbent=read(a.candidate),read(a.incumbent)
    if a.second_seeds:candidate,incumbent=[candidate,read(a.second_seeds[0])],[incumbent,read(a.second_seeds[1])]
    print(json.dumps(assess(candidate,incumbent,read(a.anchor)),indent=2))


if __name__=='__main__':main()
