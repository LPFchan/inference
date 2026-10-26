"""Decode CSF-compressed NVFP4 expert scales in the default loader.

Port of vllm-project/vllm#60646's loader hook. The decoder lives in
``vllm_csf_scales``; the hook rewrites each stream pair into its ``<name>``
E4M3 scale before the model sees the weights.
"""

import sysconfig
from pathlib import Path

loader = (
    Path(sysconfig.get_paths()["purelib"])
    / "vllm/model_executor/model_loader/default_loader.py"
)
old = """        # Apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)
"""
new = """        # Apply the prefix.
        from vllm_csf_scales import decode_csf_scale_streams

        return (
            (source.prefix + name, tensor)
            for (name, tensor) in decode_csf_scale_streams(weights_iterator)
        )
"""
source = loader.read_text()
if source.count(old) != 1:
    raise RuntimeError(f"expected one CSF loader anchor in {loader}, found {source.count(old)}")
loader.write_text(source.replace(old, new, 1))
print(f"installed CSF scale decoding in {loader}")
