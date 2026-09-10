"""Fail-closed transformations of NVIDIA's pinned dense CuTe DSL source."""


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Pinned dense source contract changed: {old[:80]!r}")
    return text.replace(old, new)


def adapt_dense_source(text):
    replacements = [
        ("from common import (", "from .dense_common import ("),
        ("        self.acc_dtype = acc_dtype",
         "        self.tile_uniform_alpha = True\n        self.acc_dtype = acc_dtype"),
        ("        c: cute.Tensor,\n        max_active_clusters:",
         "        c: cute.Tensor,\n        alpha_ptr: cute.Pointer,\n        max_active_clusters:"),
        ("            tile_sched_params,\n        ).launch(",
         "            tile_sched_params,\n            alpha_ptr,\n        ).launch("),
        ("        c_ptr: cute.Pointer,\n        m:",
         "        c_ptr: cute.Pointer,\n        alpha_ptr: cute.Pointer,\n        m:"),
        ("return self(a, b, sfa, sfb, c, max_active_clusters, stream)",
         "return self(a, b, sfa, sfb, c, alpha_ptr, max_active_clusters, stream)"),
        ("        tile_sched_params: utils.PersistentTileSchedulerParams,\n    ):",
         "        tile_sched_params: utils.PersistentTileSchedulerParams,\n        alpha_ptr: cute.Pointer,\n    ):"),
        ("        tCgC = thr_mma.partition_C(gC_mnl)",
         """        tCgC = thr_mma.partition_C(gC_mnl)
        # Broadcast checkpoint-global products along M, keeping the same
        # coordinate partition as the accumulator. N includes zero padding.
        mAlpha = cute.make_tensor(
            alpha_ptr, cute.make_layout(mC_mnl.shape, stride=(0, 1, 0)))
        gAlpha = cute.local_tile(
            mAlpha, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        tCgAlpha = thr_mma.partition_C(gAlpha)"""),
        ("                tTR_rAcc,\n            ) = self.epilog_tmem_copy_and_partition(\n                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs,",
         "                tTR_rAcc,\n                tTR_gAlpha,\n            ) = self.epilog_tmem_copy_and_partition(\n                epi_tidx, tCtAcc_base, tCgAlpha, epi_tile, use_2cta_instrs,"),
        ("        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc",
         "        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_gC"),
        ("                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])",
         """                if cutlass.const_expr(self.tile_uniform_alpha):
                    tile_alpha = mAlpha[0, mma_tile_coord_mnl[1] * self.mma_tiler_mn[1], 0]
                else:
                    tTR_alpha = tTR_gAlpha[(None, None, None, None, None, *mma_tile_coord_mnl)]
                    tTR_alpha = cute.group_modes(tTR_alpha, 3, cute.rank(tTR_alpha))
                    rAlpha = cute.make_rmem_tensor(tTR_rAcc.shape, cutlass.Float32)
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])"""),
        ("                    tRS_rC.store(acc_vec.to(self.c_dtype))",
         """                    if cutlass.const_expr(self.tile_uniform_alpha):
                        tRS_rC.store((acc_vec * tile_alpha).to(self.c_dtype))
                    else:
                        alpha_mn = tTR_alpha[(None, None, None, subtile_idx)]
                        for idx in cutlass.range_constexpr(cute.size(rAlpha)):
                            rAlpha[idx] = alpha_mn[idx]
                        alpha_vec = tiled_copy_r2s.retile(rAlpha).load()
                        tRS_rC.store((acc_vec * alpha_vec).to(self.c_dtype))"""),
    ]
    for old, new in replacements:
        text = replace_once(text, old, new)
    # Only the kernel class is imported by our Torch wrapper; NVIDIA's CLI
    # allocation/export helpers retain their original FP16/FP8 contracts.
    return text
