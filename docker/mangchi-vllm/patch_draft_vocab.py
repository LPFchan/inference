"""Reduced-vocabulary drafting for the Qwen4Exp MTP head (FR-Spec).

The draft head's output projection is the one big BF16 tensor left in an
otherwise NVFP4 model: lm_head is [248320, 2560] = 1.27 GB, and a draft pass
reads all of it to score every candidate token. Measured on this checkpoint
that projection is ~98% of a draft pass's weight traffic (the routed experts
are 4-bit with a 640 intermediate, so they contribute ~0.03 GB), and at k=4
the four draft passes dominate the decode step. Restricting the draft to the
first QWEN4EXP_DRAFT_VOCAB token ids and masking the rest to -inf cuts that
read by (1 - K/V).

Token id is the frequency proxy: the tokenizer is BPE, so merge order puts
frequent tokens at low ids and no corpus is needed. Measured coverage over
8,110 tokens this model actually generated: 97.89% below 65536 and 99.96%
below 98304 (prose is the worst case at 96.76% / 65536).

Correctness is unaffected. The target model still verifies every draft over
the full vocabulary, so a token outside the draft set is simply never proposed
and the draft is rejected at that position. Speculative decoding's acceptance
rule is untouched, so the emitted text is what the unaccelerated model would
have produced; only draft acceptance can move.

Idea: FR-Spec, frequency-ranked speculative sampling (Zhao et al. 2025).
Approach follows tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark's
single-spark-vllm-tp1/patch/mtp_draft_vocab.py (Apache-2.0), ported to this
pin, which carries the same four-line compute_logits.
"""

import sysconfig
from pathlib import Path

site_packages = Path(sysconfig.get_paths()["purelib"])
mtp = site_packages / "vllm/models/qwen4_exp/nvidia/mtp.py"

source = mtp.read_text()

old = (
    "    def compute_logits(\n"
    "        self, hidden_states: torch.Tensor, spec_step_idx: int = 0\n"
    "    ) -> torch.Tensor | None:\n"
    "        return self.logits_processor(self.lm_head, hidden_states)\n"
)
new = (
    "    def compute_logits(\n"
    "        self, hidden_states: torch.Tensor, spec_step_idx: int = 0\n"
    "    ) -> torch.Tensor | None:\n"
    "        k = _draft_vocab_size()\n"
    "        if k <= 0 or not hasattr(self.lm_head, \"weight\"):\n"
    "            return self.logits_processor(self.lm_head, hidden_states)\n"
    "        return self._reduced_vocab_logits(hidden_states, k)\n"
    "\n"
    "    def _reduced_vocab_logits(\n"
    "        self, hidden_states: torch.Tensor, k: int\n"
    "    ) -> torch.Tensor:\n"
    "        \"\"\"Logits for token ids [0, k) only; every other id is -inf.\n"
    "\n"
    "        Each TP rank owns a vocab slice of lm_head. It multiplies the rows of\n"
    "        its slice falling inside [0, k) into a [n, k] buffer and the ranks sum,\n"
    "        since each column is owned by exactly one rank. The result is scattered\n"
    "        into a full-width [n, vocab] tensor so the speculator's argmax still\n"
    "        yields global token ids.\n"
    "        \"\"\"\n"
    "        import torch.nn.functional as F\n"
    "\n"
    "        from vllm.distributed import (\n"
    "            get_tensor_model_parallel_world_size,\n"
    "            tensor_model_parallel_all_reduce,\n"
    "        )\n"
    "\n"
    "        head = self.lm_head\n"
    "        cache = getattr(self, \"_draft_vocab_cache\", None)\n"
    "        if cache is None or cache[0] != k:\n"
    "            start = int(head.shard_indices.org_vocab_start_index)\n"
    "            end = int(head.shard_indices.org_vocab_end_index)\n"
    "            lo, hi = max(start, 0), min(end, k)\n"
    "            w = (\n"
    "                head.weight[lo - start : hi - start].contiguous()\n"
    "                if hi > lo\n"
    "                else None\n"
    "            )\n"
    "            cache = (k, start, lo, hi, w)\n"
    "            self._draft_vocab_cache = cache\n"
    "        _, start, lo, hi, w = cache\n"
    "        n = hidden_states.shape[0]\n"
    "        small = torch.zeros(\n"
    "            (n, k), dtype=hidden_states.dtype, device=hidden_states.device\n"
    "        )\n"
    "        if w is not None:\n"
    "            small[:, lo:hi] = F.linear(hidden_states.to(w.dtype), w).to(small.dtype)\n"
    "        if get_tensor_model_parallel_world_size() > 1:\n"
    "            small = tensor_model_parallel_all_reduce(small)\n"
    "        vocab = int(self.config.vocab_size)\n"
    "        full = torch.full(\n"
    "            (n, vocab),\n"
    "            float(\"-inf\"),\n"
    "            dtype=small.dtype,\n"
    "            device=small.device,\n"
    "        )\n"
    "        full[:, :k] = small\n"
    "        scale = getattr(self.logits_processor, \"scale\", 1.0)\n"
    "        if scale != 1.0:\n"
    "            full.mul_(scale)\n"
    "        return full\n"
)
if old not in source:
    raise RuntimeError(f"Qwen4Exp MTP compute_logits not found in {mtp}")
source = source.replace(old, new, 1)

helper = (
    "\n"
    "def _draft_vocab_size() -> int:\n"
    "    \"\"\"Draft-vocabulary row count from QWEN4EXP_DRAFT_VOCAB; 0 disables.\"\"\"\n"
    "    import os\n"
    "\n"
    "    try:\n"
    "        return int(os.environ.get(\"QWEN4EXP_DRAFT_VOCAB\", \"0\"))\n"
    "    except ValueError:\n"
    "        return 0\n"
    "\n"
    "\n"
)
anchor = "\nclass Qwen4ExpMultiTokenPredictor("
if anchor not in source:
    anchor = "\nclass Qwen4ExpMTP("
    if anchor not in source:
        raise RuntimeError(f"no MTP class anchor to place the helper in {mtp}")
source = source.replace(anchor, helper + anchor.lstrip("\n"), 1)

mtp.write_text(source)
print(f"patched {mtp}")
