"""Make the disk-backed PLE gather a piecewise CUDA-graph split point.

`vllm_ple_mmap.py` replaces the PLE n-gram embedding with a host gather: it
copies ids to CPU, reads rows off NVMe, and stages them back to the GPU. A host
round trip cannot run inside CUDA graph capture, which is why this model is
served with `--enforce-eager` and gets no graph replay at all.

vLLM already has the mechanism for this. `CompilationConfig._attention_ops`
lists operations that torch.compile must treat as opaque and that terminate a
piecewise graph; it already carries `vllm::qwen4_exp_compute_ple_ngram_ids`.
Registering the gather as a custom op and adding it to that list lets decode run
as piecewise CUDA graphs with the table still on disk: the graph splits around
the gather instead of refusing to capture.

Opt-in with QWEN4EXP_PLE_GRAPH_SPLIT=1; unset keeps the direct call so the
eager path is bit-for-bit what it was. Requires VLLM_PLE_MMAP=1 to matter.

Approach follows tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark's
single-spark-vllm-tp1 patches (ple_layer.diff registers the same op name;
compilation.diff adds the same one line), ported to this loader. Their staged
variant instead hoists the gather into the model state's prepare_inputs, which
additionally enables FULL_DECODE_ONLY; that is a larger change and is tracked
separately.
"""

import sysconfig
from pathlib import Path

site_packages = Path(sysconfig.get_paths()["purelib"])
compilation = site_packages / "vllm/config/compilation.py"
ple_mmap = site_packages / "vllm_ple_mmap.py"

# 1. Declare the gather a split point for piecewise compilation.
source = compilation.read_text()
anchor = '        # Qwen4Exp\'s AMD backend still uses these splitting ops.\n'
addition = (
    '        "vllm::qwen4_exp_ple_mmap_gather",  # host gather: stays outside '
    "piecewise graphs\n"
)
if anchor not in source:
    raise RuntimeError(f"_attention_ops PLE anchor not found in {compilation}")
if addition not in source:
    compilation.write_text(source.replace(anchor, anchor + addition, 1))

# 2. Register the gather as an opaque custom op and route forward through it.
source = ple_mmap.read_text()
old = (
    "    def patched_forward(self, input_ids, query_start_loc, ngram_context):\n"
    "        ids = self.compute_ngram_ids(\n"
    "            input_ids, query_start_loc, ngram_context\n"
    "        )\n"
    "        return self.ngram_embedding.gather(ids).flatten(-2)\n"
)
new = (
    "    def patched_forward(self, input_ids, query_start_loc, ngram_context):\n"
    "        ids = self.compute_ngram_ids(\n"
    "            input_ids, query_start_loc, ngram_context\n"
    "        )\n"
    "        if not _graph_split_enabled():\n"
    "            return self.ngram_embedding.gather(ids).flatten(-2)\n"
    "        embedding = self.ngram_embedding\n"
    "        name = getattr(self, '_ple_mmap_prefix', None)\n"
    "        if name is None or name not in _PLE_GATHER_LAYERS:\n"
    "            return embedding.gather(ids).flatten(-2)\n"
    "        out = torch.empty(\n"
    "            (ids.shape[0], ids.shape[1] * embedding.embedding_dim),\n"
    "            dtype=embedding.gather_dtype,\n"
    "            device=ids.device,\n"
    "        )\n"
    "        torch.ops.vllm.qwen4_exp_ple_mmap_gather(ids, out, name)\n"
    "        return out\n"
)
if old not in source:
    raise RuntimeError(f"patched_forward not found in {ple_mmap}")
source = source.replace(old, new, 1)

# gather_dtype: the staged rows' dtype, or the placeholder dtype before load.
old_gather = "    def gather(self, ids: torch.Tensor) -> torch.Tensor:\n"
new_gather = (
    "    @property\n"
    "    def gather_dtype(self) -> torch.dtype:\n"
    "        return (\n"
    "            self.table.torch_dtype if self.table is not None"
    " else self._zeros_dtype\n"
    "        )\n"
    "\n"
    "    def gather(self, ids: torch.Tensor) -> torch.Tensor:\n"
)
if old_gather not in source:
    raise RuntimeError(f"gather method not found in {ple_mmap}")
source = source.replace(old_gather, new_gather, 1)

# Register the module in a name -> layer map so the op can find it, and define
# the op itself. Appended at import time so the op exists before model build.
registry = '''

_PLE_GATHER_LAYERS: dict[str, object] = {}


def _graph_split_enabled() -> bool:
    import os

    return os.environ.get("QWEN4EXP_PLE_GRAPH_SPLIT", "0") == "1"


def _qwen4_exp_ple_mmap_gather(
    ids: torch.Tensor, out: torch.Tensor, layer_name: str
) -> None:
    """Host gather into a caller-owned buffer; opaque to torch.compile."""
    layer = _PLE_GATHER_LAYERS[layer_name]
    out.copy_(layer.ngram_embedding.gather(ids).flatten(-2))


def _qwen4_exp_ple_mmap_gather_fake(
    ids: torch.Tensor, out: torch.Tensor, layer_name: str
) -> None:
    return


def _register_ple_gather_op() -> None:
    if getattr(_register_ple_gather_op, "_done", False):
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="qwen4_exp_ple_mmap_gather",
        op_func=_qwen4_exp_ple_mmap_gather,
        mutates_args=["out"],
        fake_impl=_qwen4_exp_ple_mmap_gather_fake,
    )
    _register_ple_gather_op._done = True


_register_ple_gather_op()
'''
if "_PLE_GATHER_LAYERS" not in source.split("def patched_forward")[0]:
    source = source + registry

# Populate the registry when the PLE layer records its prefix.
old_prefix = "        self._ple_mmap_prefix = prefix\n"
new_prefix = (
    "        self._ple_mmap_prefix = prefix\n"
    "        _PLE_GATHER_LAYERS[prefix] = self\n"
)
if old_prefix not in source:
    raise RuntimeError(f"prefix assignment not found in {ple_mmap}")
source = source.replace(old_prefix, new_prefix, 1)

ple_mmap.write_text(source)
print(f"patched {compilation} and {ple_mmap}")
