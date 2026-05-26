# Vulkan Q1_0_g128 Support Plan

## Goal

Add `GGML_TYPE_Q1_0_g128` support to the Vulkan backend so `llama-server` can keep `Q1_0_g128` tensors on Vulkan for the same operation families already supported by CPU, CUDA, and Metal.

## Current State

- CPU, CUDA, and Metal already have explicit `Q1_0_g128` support.
- Vulkan has no `GGML_TYPE_Q1_0_g128` entries in its C++ dispatch and pipeline registration tables.
- Vulkan shader code is partially prepared for a 128-element 1-bit format already:
  - shader-side `block_q1_0` in `ggml/src/ggml-vulkan/vulkan-shaders/types.glsl` is a 128-element block.
  - coopmat2 flash-attention already contains a hardcoded path for numeric type `41`, which is `GGML_TYPE_Q1_0_g128`.
- Because the C++ Vulkan backend never wires up `GGML_TYPE_Q1_0_g128`, the runtime falls back away from meaningful Vulkan execution.

## Main Finding

The missing work is mostly backend plumbing, not new quant math.

The only non-trivial design choice is whether to:

1. keep Vulkan's current shader-side `Q1_0` naming and alias `Q1_0_g128` onto it, or
2. add an explicit shader-side `Q1_0_g128` type and make the mapping unambiguous.

Recommended approach: option 2. The shader-side Vulkan `Q1_0` layout already matches `Q1_0_g128`, not the host-side `GGML_TYPE_Q1_0` layout, so making the `g128` path explicit is safer and easier to reason about.

## Reference Implementations

- CPU reference layout and quant/dequant:
  - `ggml/src/ggml-common.h`
  - `ggml/src/ggml-quants.c`
  - `ggml/src/ggml-cpu/quants.c`
- CUDA reference:
  - `ggml/src/ggml-cuda/dequantize.cuh`
  - `ggml/src/ggml-cuda/mmq.cuh`
  - `ggml/src/ggml-cuda/convert.cu`
- Metal reference:
  - `ggml/src/ggml-metal/ggml-metal.metal`
  - `ggml/src/ggml-metal/ggml-metal-ops.cpp`

## Patch Plan

### 1. Add explicit Vulkan shader-side `Q1_0_g128` type support

Modify `ggml/src/ggml-vulkan/vulkan-shaders/types.glsl`:

- Add `QUANT_K_Q1_0_G128` and `QUANT_R_Q1_0_G128`.
- Add `struct block_q1_0_g128` with the same layout as host-side `block_q1_0_g128`:
  - `float16_t d`
  - `uint8_t qs[16]`
- Add `#if defined(DATA_A_Q1_0_G128)` mapping:
  - `QUANT_K`
  - `QUANT_R`
  - `QUANT_AUXF`
  - `A_TYPE`

Why:

- The current Vulkan shader-side `block_q1_0` is already a 128-element structure.
- Making `Q1_0_g128` explicit avoids continuing the current type/layout mismatch.

### 2. Add dequant helpers for `Q1_0_g128`

Modify `ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl`:

- Add a `#if defined(DATA_A_Q1_0_G128)` block mirroring `DATA_A_Q1_0`:
  - `vec2 dequantize(...)`
  - `vec4 dequantize4(...)`
  - `vec2 get_dm(...)`

Implementation notes:

- The bit interpretation should match CUDA and Metal `Q1_0_g128`:
  - bit set -> `+d`
  - bit clear -> `-d`
- The logic is mechanically the same as `Q1_0`, but indexed over a 128-element block.

### 3. Add coopmat2 decode helpers for `Q1_0_g128`

Modify `ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs_cm2.glsl`:

- Add `decodeBufQ1_0_g128`.
- Add `dequantFuncQ1_0_g128(...)`.
- Add `#if defined(DATA_A_Q1_0_G128)` mapping to select that decode function.

Modify `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp`:

- Replace the current hardcoded `case 41u` handling that calls `dequantFuncQ1_0(...)` with explicit `Q1_0_g128` decode.
- Keep the numeric type `41u` mapping aligned with `GGML_TYPE_Q1_0_g128` from `ggml/include/ggml.h`.

Why:

- The flash-attention shader already has a partial `type 41` path.
- That path should become explicit and consistent with the new shader-side `Q1_0_g128` type.

### 4. Extend the Vulkan shader generator to emit `q1_0_g128` variants

Modify `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`:

- Add `"q1_0_g128"` to `type_names`.
- Update the `load_vec_quant` selection block so `q1_0_g128` gets the same quant load width as `q1_0`.
- Add `q1_0_g128` to the `copy_to_quant.comp` generation loop.
- Add `q1_0_g128` to the `set_rows` generation loop.

Expected generated shader families:

- `matmul_q1_0_g128_f32`
- `matmul_q1_0_g128_f16`
- `matmul_id_q1_0_g128_f32`
- `matmul_id_subgroup_q1_0_g128_f32`
- `mul_mat_vec_q1_0_g128_f32_f32`
- `mul_mat_vec_q1_0_g128_f16_f32`
- `mul_mat_vec_id_q1_0_g128_f32`
- `dequant_q1_0_g128`
- `get_rows_q1_0_g128`
- `get_rows_q1_0_g128_f32`
- `cpy_f32_q1_0_g128`
- `cpy_q1_0_g128_f32`
- `set_rows_q1_0_g128_i32`
- `set_rows_q1_0_g128_i64`

### 5. Register `Q1_0_g128` pipelines in Vulkan initialization

Modify `ggml/src/ggml-vulkan/ggml-vulkan.cpp` in the pipeline creation section.

Add `GGML_TYPE_Q1_0_g128` entries alongside existing `GGML_TYPE_Q1_0` entries for:

- dequant + mat-vec f32/f16 pipeline creation
- dequant + mat-vec id pipeline creation
- dequant pipeline creation
- get-rows pipeline creation
- get-rows-f32 pipeline creation
- `cpy_f32_quant`
- `set_rows`
- `cpy_quant_f32`
- matmul pipeline creation
- matmul-id pipeline creation

Concrete regions to update:

- matmul f16 creation near the `CREATE_MM2(... GGML_TYPE_Q1_0 ...)` block
- matmul f32 creation near the `CREATE_MM2(GGML_TYPE_Q1_0, ...)` block
- matmul id creation near the `CREATE_MM2(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q1_0], ...)` block
- mat-vec pipeline creation where `pipeline_dequant_mul_mat_vec_*[GGML_TYPE_Q1_0]` is created
- dequant pipeline registration where `pipeline_dequant[GGML_TYPE_Q1_0]` is created
- get-rows registration where `pipeline_get_rows[GGML_TYPE_Q1_0]` is created
- get-rows-f32 registration where `pipeline_get_rows_f32[GGML_TYPE_Q1_0]` is created
- copy-to-quant registration where `pipeline_cpy_f32_quant[GGML_TYPE_Q1_0]` is created
- set-rows macro expansion where `pipeline_set_rows...[GGML_TYPE_Q1_0]` is created
- quant-to-f32 registration where `pipeline_cpy_quant_f32[GGML_TYPE_Q1_0]` is created

### 6. Add `Q1_0_g128` to Vulkan dispatch switches

Modify `ggml/src/ggml-vulkan/ggml-vulkan.cpp` switch statements so `GGML_TYPE_Q1_0_g128` is accepted and routed.

Required functions and switch blocks:

- `ggml_vk_get_to_fp16(...)`
  - add `case GGML_TYPE_Q1_0_g128`
- `ggml_vk_get_mul_mat_mat_pipeline(...)`
  - add `case GGML_TYPE_Q1_0_g128`
- `ggml_vk_get_dequantize_mul_mat_vec(...)`
  - add `case GGML_TYPE_Q1_0_g128`
- `ggml_vk_get_mul_mat_mat_id_pipeline(...)`
  - add `case GGML_TYPE_Q1_0_g128`
- `ggml_vk_get_dequantize_mul_mat_vec_id(...)`
  - add `case GGML_TYPE_Q1_0_g128`
- copy / conversion dispatch around the `src->type == GGML_TYPE_F32` and `to == GGML_TYPE_F32` switch blocks
  - add `GGML_TYPE_Q1_0_g128` to supported quant conversions

### 7. Add `Q1_0_g128` to Vulkan op support gating

Modify `ggml/src/ggml-vulkan/ggml-vulkan.cpp` in the op support function near the large `switch (op->op)` block.

Add `GGML_TYPE_Q1_0_g128` to the allowed-type switch cases for:

- `GGML_OP_MUL_MAT`
- `GGML_OP_MUL_MAT_ID`
- `GGML_OP_GET_ROWS`
- `GGML_OP_SET_ROWS`
- `GGML_OP_CONT`
- `GGML_OP_CPY`
- `GGML_OP_DUP`

For flash attention:

- update the path selector near the `Q1_0 K/V is only implemented on coopmat2` logic to also consider `GGML_TYPE_Q1_0_g128`
- update `fa_kv_ok(...)` to allow `GGML_TYPE_Q1_0_g128` under `coopmat2`

### 8. Review whether existing Vulkan `Q1_0` should be corrected or left untouched

Before finalizing the patch, decide one of these explicitly:

1. Keep existing Vulkan `Q1_0` behavior as-is and only add `Q1_0_g128` support.
2. Correct Vulkan shader-side `Q1_0` to match the true host-side `GGML_TYPE_Q1_0` layout and keep `Q1_0_g128` separate.

Recommendation for the first patch:

- Do not attempt to fix both in one change.
- Add explicit `Q1_0_g128` support first.
- Leave existing Vulkan `Q1_0` behavior alone unless testing proves it is also broken.

## Suggested Implementation Order

1. `types.glsl`
2. `dequant_funcs.glsl`
3. `dequant_funcs_cm2.glsl`
4. `flash_attn_cm2.comp`
5. `vulkan-shaders-gen.cpp`
6. regenerate / rebuild Vulkan shader outputs
7. `ggml-vulkan.cpp` pipeline creation
8. `ggml-vulkan.cpp` dispatch switches
9. `ggml-vulkan.cpp` op support gating
10. build and validate

## Validation Checklist

### Build Validation

- Rebuild Vulkan shaders and `llama-server`.
- Verify generated symbols for `q1_0_g128` exist in the Vulkan shader output.

### Functional Validation

- Confirm a `Q1_0_g128` model no longer falls back at Vulkan op selection.
- Run `llama-server --list-devices` and start a `Q1_0_g128` model with `--metrics --perf`.
- Verify:
  - Vulkan op selection succeeds for `MUL_MAT`
  - flash attention uses coopmat2 path when applicable
  - no immediate CPU fallback for `GET_ROWS`, `CPY`, `SET_ROWS`, or dequant

### Runtime Validation

- Repeat the earlier live test with Windows counters:
  - process CPU percent
  - GPU engine utilization percent
- Expected improvement:
  - materially higher GPU compute utilization
  - materially lower CPU saturation
  - tokens/sec above the earlier `~1.05 tok/s` result

### Regression Checks

- Verify regular Vulkan `Q1_0` models still load and run.
- If possible, test one `Q1_0` and one `Q1_0_g128` model back to back.

## Risk Notes

- The biggest risk is the existing Vulkan shader-side `Q1_0` layout mismatch.
- Flash attention already appears partially wired for type `41`, so patching only the high-level C++ tables may expose latent assumptions in shader naming.
- Keep the first patch narrow: `Q1_0_g128` enablement only, with no broad refactor of legacy `Q1_0` unless a validation failure forces it.

## Expected Scope

This should be a medium-sized Vulkan backend patch, mostly mechanical.

Files expected to change:

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- `ggml/src/ggml-vulkan/vulkan-shaders/types.glsl`
- `ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl`
- `ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs_cm2.glsl`
- `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp`
- `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`
- generated Vulkan shader output files produced by the build
