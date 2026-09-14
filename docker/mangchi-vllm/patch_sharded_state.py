"""Load prepared Thor MoE state directly and honor MTP's draft loader config.

vLLM's sharded-state format records parameters after quantization backends have
prepared them. The Thor CuTe DSL backend changes several MoE tensor shapes, so
the stock loader cannot copy those tensors into a freshly constructed model.
PLE mmap tensors are intentionally supplied by the original checkpoint and
must not be restored from the sharded target checkpoint.
"""

import sysconfig
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if source.count(old) != 1:
        raise RuntimeError(f"expected one patch anchor in {path}, found {source.count(old)}")
    path.write_text(source.replace(old, new, 1))


site_packages = Path(sysconfig.get_paths()["purelib"])
loader = site_packages / "vllm/model_executor/model_loader/sharded_state_loader.py"
eagle = site_packages / "vllm/v1/worker/gpu/spec_decode/eagle/utils.py"

replace_once(
    loader,
    """        state_dict = self._filter_subtensors(model.state_dict())
        counter_before_loading_weights = time.perf_counter()
        for key, tensor in self.iterate_over_files(filepaths):
            # If loading with LoRA enabled, additional padding may
""",
    """        state_dict = self._filter_subtensors(model.state_dict())
        model_parameters = dict(model.named_parameters())
        counter_before_loading_weights = time.perf_counter()
        for key, tensor in self.iterate_over_files(filepaths):
            ple_mmap_key = ".ple.ple_embedding.ngram_embedding." in key
            if (
                key not in state_dict
                and os.environ.get("VLLM_PLE_MMAP") == "1"
                and ple_mmap_key
            ):
                continue
            # If loading with LoRA enabled, additional padding may
""",
)

replace_once(
    loader,
    """            param_data = state_dict[key].data
            param_shape = state_dict[key].shape
            for dim, size in enumerate(tensor.shape):
""",
    """            param_data = state_dict[key].data
            param_shape = state_dict[key].shape
            thor_prepared = (
                os.environ.get("VLLM_THOR_CUTEDSL_MOE") == "1"
                and ".experts.routed_experts." in key
                and key.rsplit(".", 1)[-1]
                in {
                    "w13_weight",
                    "w2_weight",
                    "w13_weight_scale",
                    "w2_weight_scale",
                    "w13_weight_scale_2",
                    "w13_input_scale",
                    "w2_weight_scale_2",
                    "w2_input_scale",
                }
                and tensor.shape != param_shape
            )
            if thor_prepared:
                model_parameters[key].data = tensor.to(
                    device=model_parameters[key].device,
                    non_blocking=False,
                )
                state_dict.pop(key)
                continue
            for dim, size in enumerate(tensor.shape):
""",
)

replace_once(
    loader,
    """        if state_dict:
            raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")
""",
    """        if os.environ.get("VLLM_PLE_MMAP") == "1":
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if ".ple.ple_embedding.ngram_embedding." not in key
            }
        if state_dict:
            raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")
""",
)

# MTP/EAGLE accidentally inherited the target's load format at this pin. Pass
# the already-supported draft_load_config, as the Gemma4 speculator does.
replace_once(
    eagle,
    """        eagle_model = get_model(
            vllm_config=vllm_config, model_config=draft_model_config
        )
""",
    """        eagle_model = get_model(
            vllm_config=vllm_config,
            model_config=draft_model_config,
            load_config=speculative_config.draft_load_config,
        )
""",
)

print(f"patched {loader} and {eagle}")
