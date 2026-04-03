// Q1_0 GEMM Kernel - Direct GGML layout (no transpose needed)
// Each work-item computes an 8x4 output tile
// gy indexes 8 output rows (N dimension - batch/sequence)
// gx indexes 4 output columns (M dimension - output features)
//
// GGML stores B as N rows of K elements: B[n][k] at index n*K + k
// This kernel loads B values with strided access to avoid transpose

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#ifndef REQD_SUBGROUP_SIZE_128
#define REQD_SUBGROUP_SIZE_128
#endif

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif

kernel void kernel_mul_mat_q1_0_Ab_Bi_8x4(
        global const uchar * src0_q,        // packed 1-bit weights (SOA: q buffer)
        global const half  * src0_d,        // scales (SOA: d buffer)
        global const uchar * src1_base,     // B activations base pointer
        ulong src1_offset,                  // offset into src1 buffer
        global uchar * dst_base,            // output base pointer
        ulong dst_offset,                   // offset into dst buffer
        int m,                              // M (output features / rows of A)
        int n,                              // N (batch size)
        int k,                              // K (input features / cols of A)
        int n_no_padding                    // N without padding (for bounds check)
) {
    // Apply offsets
    global const float * src1 = (global const float *)(src1_base + src1_offset);
    global float * dst = (global float *)(dst_base + dst_offset);
    int gy = get_global_id(0);  // output row tile (0 to N/8)
    int gx = get_global_id(1);  // output column tile (0 to M/4)
    int gx_4 = gx << 2;         // starting column (gx * 4)

    float8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;  // 8x4 output tile

    int num_blocks = k / 32;
    int row_base = gy << 3;  // gy * 8 = starting output row

    // Pointers for 4 weight columns (SOA layout, row-major)
    // For row r, block b: offset = (r * num_blocks + b) * 4 bytes
    global const uint* weight_ptr0 = (global const uint*)(src0_q + (gx_4 + 0) * num_blocks * 4);
    global const uint* weight_ptr1 = (global const uint*)(src0_q + (gx_4 + 1) * num_blocks * 4);
    global const uint* weight_ptr2 = (global const uint*)(src0_q + (gx_4 + 2) * num_blocks * 4);
    global const uint* weight_ptr3 = (global const uint*)(src0_q + (gx_4 + 3) * num_blocks * 4);

    // Scale pointers for 4 columns
    global const half* scale_ptr0 = src0_d + (gx_4 + 0) * num_blocks;
    global const half* scale_ptr1 = src0_d + (gx_4 + 1) * num_blocks;
    global const half* scale_ptr2 = src0_d + (gx_4 + 2) * num_blocks;
    global const half* scale_ptr3 = src0_d + (gx_4 + 3) * num_blocks;

    for (int block = 0; block < num_blocks; block++) {
        // Load scales for 4 columns
        float s0 = (float)scale_ptr0[block];
        float s1 = (float)scale_ptr1[block];
        float s2 = (float)scale_ptr2[block];
        float s3 = (float)scale_ptr3[block];

        // Load 32 bits for 4 columns (each uint has 32 weight bits)
        uint bits0 = weight_ptr0[block];
        uint bits1 = weight_ptr1[block];
        uint bits2 = weight_ptr2[block];
        uint bits3 = weight_ptr3[block];

        // Process 32 K elements in this block
        int k_base = block * 32;

        #pragma unroll 4
        for (int i = 0; i < 32; i++) {
            int k_idx = k_base + i;

            // Load 8 B values for 8 output rows at K position k_idx
            // GGML layout: B[n][k] at index n*K + k (N rows of K elements)
            // We need B[row_base+0..7][k_idx]
            float8 B;
            B.s0 = (row_base + 0 < n) ? src1[(row_base + 0) * k + k_idx] : 0.0f;
            B.s1 = (row_base + 1 < n) ? src1[(row_base + 1) * k + k_idx] : 0.0f;
            B.s2 = (row_base + 2 < n) ? src1[(row_base + 2) * k + k_idx] : 0.0f;
            B.s3 = (row_base + 3 < n) ? src1[(row_base + 3) * k + k_idx] : 0.0f;
            B.s4 = (row_base + 4 < n) ? src1[(row_base + 4) * k + k_idx] : 0.0f;
            B.s5 = (row_base + 5 < n) ? src1[(row_base + 5) * k + k_idx] : 0.0f;
            B.s6 = (row_base + 6 < n) ? src1[(row_base + 6) * k + k_idx] : 0.0f;
            B.s7 = (row_base + 7 < n) ? src1[(row_base + 7) * k + k_idx] : 0.0f;

            // Dequantize 4 weights (one per column)
            // bit=1 -> +scale, bit=0 -> -scale
            float w0 = ((bits0 >> i) & 1u) ? s0 : -s0;
            float w1 = ((bits1 >> i) & 1u) ? s1 : -s1;
            float w2 = ((bits2 >> i) & 1u) ? s2 : -s2;
            float w3 = ((bits3 >> i) & 1u) ? s3 : -s3;

            // Accumulate: each c is 8 values for 8 output rows
            c0 += B * w0;
            c1 += B * w1;
            c2 += B * w2;
            c3 += B * w3;
        }
    }

    // Write 8x4 tile to output
    // Output layout: row-major, C[row][col] at dst[row * m + col]
    if (row_base + 0 < n_no_padding) {
        vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + (row_base + 0) * m + (gx << 2));
    }
    if (row_base + 1 < n_no_padding) {
        vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + (row_base + 1) * m + (gx << 2));
    }
    if (row_base + 2 < n_no_padding) {
        vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + (row_base + 2) * m + (gx << 2));
    }
    if (row_base + 3 < n_no_padding) {
        vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + (row_base + 3) * m + (gx << 2));
    }
    if (row_base + 4 < n_no_padding) {
        vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + (row_base + 4) * m + (gx << 2));
    }
    if (row_base + 5 < n_no_padding) {
        vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + (row_base + 5) * m + (gx << 2));
    }
    if (row_base + 6 < n_no_padding) {
        vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + (row_base + 6) * m + (gx << 2));
    }
    if (row_base + 7 < n_no_padding) {
        vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + (row_base + 7) * m + (gx << 2));
    }
}
