//------------------------------------------------------------------------------
// Q1_0 8x Flat Kernel for Token Generation (Adreno GPU)
//
// Q1_0 format:
//   - Block size: 32 elements (QK1_0 = 32)
//   - Storage: 4 bytes quants (32 bits) + 2 bytes scale (fp16) = 6 bytes/block
//   - Dequantization: bit ? d : -d  (equivalently: d * (2*bit - 1))
//   - Layout: SOA (separate q and d buffers)
//
// This kernel processes 8 rows per subgroup (N_DST=8) for token generation.
//------------------------------------------------------------------------------

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#define QK1_0 32  // Elements per block

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

//------------------------------------------------------------------------------
// Q1_0 Dot Product Function
//
// Computes dot product of one Q1_0 block (32 weights) with 32 float activations.
// Each thread processes a subset of elements based on its position in the subgroup.
//
// Parameters:
//   x  - Pointer to 4 bytes (32 bits) of quantized weights for this block
//   dh - Pointer to fp16 scale for this block
//   yl - 16 float values from activation vector (subset this thread handles)
//   il - Offset within block (0 or 16) based on thread's position
//
// For Q1_0 (no shuffling):
//   - Bits are contiguous: bit[i] corresponds to weight[i]
//   - il=0: thread handles elements 0-15
//   - il=16: thread handles elements 16-31
//------------------------------------------------------------------------------
inline float block_q1_0_dot_y_flat(
        global uchar * x,
        global half  * dh,
        float16 yl,
        int il
) {
    float d = *dh;
    
    global uchar * qs = x;
    // Load 2 bytes (16 bits) as 2 separate bytes
    uchar b0 = qs[il/8];      // First byte (bits 0-7)
    uchar b1 = qs[il/8 + 1];  // Second byte (bits 8-15)
    
    // OPTIMIZATION: Extract bits using byte-level masking + float4 dot products
    float4 w0, w1, w2, w3;
    w0.s0 = (b0 & 0x01) ? 1.f : -1.f;
    w0.s1 = (b0 & 0x02) ? 1.f : -1.f;
    w0.s2 = (b0 & 0x04) ? 1.f : -1.f;
    w0.s3 = (b0 & 0x08) ? 1.f : -1.f;
    
    w1.s0 = (b0 & 0x10) ? 1.f : -1.f;
    w1.s1 = (b0 & 0x20) ? 1.f : -1.f;
    w1.s2 = (b0 & 0x40) ? 1.f : -1.f;
    w1.s3 = (b0 & 0x80) ? 1.f : -1.f;
    
    w2.s0 = (b1 & 0x01) ? 1.f : -1.f;
    w2.s1 = (b1 & 0x02) ? 1.f : -1.f;
    w2.s2 = (b1 & 0x04) ? 1.f : -1.f;
    w2.s3 = (b1 & 0x08) ? 1.f : -1.f;
    
    w3.s0 = (b1 & 0x10) ? 1.f : -1.f;
    w3.s1 = (b1 & 0x20) ? 1.f : -1.f;
    w3.s2 = (b1 & 0x40) ? 1.f : -1.f;
    w3.s3 = (b1 & 0x80) ? 1.f : -1.f;
    
    // Use float4 dot products (OpenCL supports up to float4)
    float4 y0 = (float4)(yl.s0, yl.s1, yl.s2, yl.s3);
    float4 y1 = (float4)(yl.s4, yl.s5, yl.s6, yl.s7);
    float4 y2 = (float4)(yl.s8, yl.s9, yl.sa, yl.sb);
    float4 y3 = (float4)(yl.sc, yl.sd, yl.se, yl.sf);
    
    float acc = dot(w0, y0) + dot(w1, y1) + dot(w2, y2) + dot(w3, y3);
    
    return d * acc;
}

//------------------------------------------------------------------------------
// Kernel Configuration for Adreno
//------------------------------------------------------------------------------
#define N_DST 8        // Each subgroup produces 8 output values
#define N_SIMDGROUP 1  // One subgroup per workgroup
#define N_SIMDWIDTH 64 // Adreno subgroup size

//------------------------------------------------------------------------------
// Main Kernel Logic
//
// Work distribution:
//   - Each workgroup contains one 64-wide subgroup
//   - Each subgroup computes 8 consecutive output rows (N_DST=8)
//   - Within subgroup: threads are paired (tid/2 gives block index, tid%2 gives half)
//   - 64 threads / 2 = 32 threads work on blocks in parallel
//   - Each pair of threads handles one block (one thread per 16 elements)
//------------------------------------------------------------------------------
inline void mul_vec_q1_0_f32_8x_flat(
        global uchar * src0_q,
        global half  * src0_d,
        global float * src1,
        global float * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    // Number of blocks per row
    const ulong nb = ne00 / QK1_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    // First output row for this subgroup
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im % ne12;
    int i13 = im / ne12;

    // Offset calculations for SOA layout
    // Scales buffer: one fp16 per block
    ulong offset0_d = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    // Quants buffer: 4 bytes (32 bits) per block
    ulong offset0_q = (first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02)) * (QK1_0/8);

    global uchar * x = src0_q + offset0_q;
    global half  * d = src0_d + offset0_d;
    global float * y = src1 + r1*ne10 + im*ne00*ne1;

    float16 yl;
    float8 sumf = 0.f;

    // Thread assignment within subgroup:
    // ix = thread's block index (0-31 for 64 threads paired)
    // il = which half of the 32-element block (0 or 16)
    int ix = get_sub_group_local_id() / 2;
    int il = 16 * (get_sub_group_local_id() % 2);

    // Pointer to this thread's portion of y
    global float * yb = y + ix * QK1_0 + il;

    // Process all blocks, stride by number of block-pairs per iteration
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/2) {
        yl.s0 = yb[0];
        yl.s1 = yb[1];
        yl.s2 = yb[2];
        yl.s3 = yb[3];
        yl.s4 = yb[4];
        yl.s5 = yb[5];
        yl.s6 = yb[6];
        yl.s7 = yb[7];
        yl.s8 = yb[8];
        yl.s9 = yb[9];
        yl.sa = yb[10];
        yl.sb = yb[11];
        yl.sc = yb[12];
        yl.sd = yb[13];
        yl.se = yb[14];
        yl.sf = yb[15];

        sumf.s0 += block_q1_0_dot_y_flat(x + ib*(QK1_0/8) + 0*nb*(QK1_0/8), d + ib + 0*nb, yl, il);
        sumf.s1 += block_q1_0_dot_y_flat(x + ib*(QK1_0/8) + 1*nb*(QK1_0/8), d + ib + 1*nb, yl, il);
        sumf.s2 += block_q1_0_dot_y_flat(x + ib*(QK1_0/8) + 2*nb*(QK1_0/8), d + ib + 2*nb, yl, il);
        sumf.s3 += block_q1_0_dot_y_flat(x + ib*(QK1_0/8) + 3*nb*(QK1_0/8), d + ib + 3*nb, yl, il);
        sumf.s4 += block_q1_0_dot_y_flat(x + ib*(QK1_0/8) + 4*nb*(QK1_0/8), d + ib + 4*nb, yl, il);
        sumf.s5 += block_q1_0_dot_y_flat(x + ib*(QK1_0/8) + 5*nb*(QK1_0/8), d + ib + 5*nb, yl, il);
        sumf.s6 += block_q1_0_dot_y_flat(x + ib*(QK1_0/8) + 6*nb*(QK1_0/8), d + ib + 6*nb, yl, il);
        sumf.s7 += block_q1_0_dot_y_flat(x + ib*(QK1_0/8) + 7*nb*(QK1_0/8), d + ib + 7*nb, yl, il);

        yb += QK1_0 * (N_SIMDWIDTH/2);
    }

    // Reduce across subgroup - sum contributions from all threads
    float8 tot = (float8)(
        sub_group_reduce_add(sumf.s0), sub_group_reduce_add(sumf.s1),
        sub_group_reduce_add(sumf.s2), sub_group_reduce_add(sumf.s3),
        sub_group_reduce_add(sumf.s4), sub_group_reduce_add(sumf.s5),
        sub_group_reduce_add(sumf.s6), sub_group_reduce_add(sumf.s7)
    );

    // First thread in subgroup writes the 8 output values
    if (get_sub_group_local_id() == 0) {
        if (first_row + 0 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 0] = tot.s0;
        }
        if (first_row + 1 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 1] = tot.s1;
        }
        if (first_row + 2 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 2] = tot.s2;
        }
        if (first_row + 3 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 3] = tot.s3;
        }
        if (first_row + 4 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 4] = tot.s4;
        }
        if (first_row + 5 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 5] = tot.s5;
        }
        if (first_row + 6 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 6] = tot.s6;
        }
        if (first_row + 7 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 7] = tot.s7;
        }
    }
}

//------------------------------------------------------------------------------
// Kernel Entry Point
//------------------------------------------------------------------------------
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q1_0_f32_8x_flat(
        global uchar * src0_q,
        global half  * src0_d,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    mul_vec_q1_0_f32_8x_flat(src0_q, src0_d, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
}



