#!/bin/bash
L=$1; P=$2
M=/models/qwen3.8-flash-next-abliterated-w4a4
snap() { curl -fsS http://127.0.0.1:8002/metrics | grep -E '^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total|^vllm:spec_decode_num_accepted_tokens_per_pos_total' | awk '{print $1"="$2}'; }
snap > /tmp/$L.before
nvidia-smi dmon -s u -d 1 -c 400 > /tmp/$L.dmon 2>/dev/null &
DP=$!
S=$(date +%s.%N)
curl -fsS http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d "$(python3 -c "import json,sys; print(json.dumps({'model':'$M','messages':[{'role':'user','content':sys.argv[1]}],'temperature':0,'max_tokens':400}))" "$P")" \
  > /tmp/$L.resp
E=$(date +%s.%N)
kill $DP 2>/dev/null
snap > /tmp/$L.after
python3 - "$L" "$S" "$E" <<'PY'
import sys,json
L,S,E=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
b=dict(l.strip().rsplit('=',1) for l in open(f'/tmp/{L}.before'))
a=dict(l.strip().rsplit('=',1) for l in open(f'/tmp/{L}.after'))
def d(k,extra=None):
    kk=[x for x in a if x.startswith(k) and (extra is None or extra in x)]
    return sum(float(a[x])-float(b.get(x,0)) for x in kk)
u=json.load(open(f'/tmp/{L}.resp'))['usage']
dr=d('vllm:spec_decode_num_drafts_total'); dt=d('vllm:spec_decode_num_draft_tokens_total'); ac=d('vllm:spec_decode_num_accepted_tokens_total')
p0=d('vllm:spec_decode_num_accepted_tokens_per_pos_total','position="0"')
p1=d('vllm:spec_decode_num_accepted_tokens_per_pos_total','position="1"')
sm=[int(l.split()[1]) for l in open(f'/tmp/{L}.dmon') if l.strip() and not l.startswith('#') and l.split()[1].isdigit()]
wall=E-S
print(f'--- {L} ---')
print(f'tokens {u["completion_tokens"]}  wall {wall:.1f}s  {u["completion_tokens"]/wall:.2f} tok/s end-to-end')
print(f'accept {100*ac/dt if dt else 0:.1f}%  mean accept len {(ac+dr)/dr if dr else 0:.2f}/3.00  pos0 {100*p0/dr if dr else 0:.1f}% pos1 {100*p1/dr if dr else 0:.1f}%')
if sm: print(f'GPU sm util: n={len(sm)} mean {sum(sm)/len(sm):.0f}%  median {sorted(sm)[len(sm)//2]}%  share<50%: {100*sum(1 for x in sm if x<50)/len(sm):.0f}%  share<20%: {100*sum(1 for x in sm if x<20)/len(sm):.0f}%')
PY
