/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*!
 * \brief Top-level routines for decoding LDPC transport channels
 */

// [from gNB coding]
#include "openair1/PHY/CODING/nrLDPC_decoder/nrLDPC_types.h"
#include "openair1/PHY/CODING/nrLDPC_coding/nrLDPC_coding_interface.h"
#include "openair1/PHY/CODING/coding_defs.h"
#include "log.h"
#include "defs.h"

#include <stdalign.h>
#include <stdint.h>
#include <syscall.h>
#include <time.h>
#include <pthread.h>
#include "PHY/gpu_compat.h"
#include "PHY/gpu_alloc.h"

#include "nrLDPC_CUDA_shared_param.h"

// #define gNB_DEBUG_TRACE

#define OAI_LDPC_DECODER_MAX_NUM_LLR 27000 // 26112 // NR_LDPC_NCOL_BG1*NR_LDPC_ZMAX = 68*384
// #define DEBUG_CRC
#ifdef DEBUG_CRC
#define PRINT_CRC_CHECK(a) a
#else
#define PRINT_CRC_CHECK(a)
#endif


gpuStream_t decoderStreams[MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4];

int d_array_size = 0;

void launch_deinterleave_i16(int Qm, int E1, int E2, int C, int r_firstE2,int16_t* e, const int16_t* f,gpuStream_t *s,int8_t sidx);
int nr_rate_matching_ldpc_rx_cuda(uint32_t Tbslbrm,
                                  uint8_t BG,
                                  uint16_t Z,
                                  int16_t *d,
                                  int16_t *soft_input,
                                  int8_t *llr_buffer,
                                  uint32_t K,
                                  uint8_t C,
                                  uint8_t rvidx,
                                  uint8_t clear,
                                  uint32_t E1,
                                  uint32_t E2,
                                  uint32_t r_firstE2,
                                  uint32_t F,
                                  uint32_t Foffset,
                                  gpuStream_t *s,
                                  int8_t sidx);

extern int pageable, integrated;
extern int8_t *p_llr_dev, *p_out_dev;
int16_t **harq_d_array;
int16_t *harq_d_array_dev;
int16_t *harq_e_dev;
int16_t *harq_f_dev;
pthread_mutex_t decoder_mutex = PTHREAD_MUTEX_INITIALIZER;

extern int32_t LDPCdecoder_cuda(t_nrLDPC_dec_params *p_decParams,
                                int8_t *p_llr,
                                uint8_t *p_out,
                                t_nrLDPC_time_stats *p_profiler,
                                decode_abort_t *ab);

// LLRs already in device memory (a GPU inner receiver wrote them): the deinterleaver reads them in place
static bool llr_on_device(const void *p)
{
  gpuPointerAttributes_t a;
  if (gpuPointerGetAttributes(&a, p) != gpuSuccess) {
    (void)gpuGetLastError();
    return false;
  }
  return a.type == gpuMemoryTypeDevice;
}

const uint8_t *ldpc_cuda_seg_ok(void);
static uint8_t *group_out;
const int nr_ldpc_cuda_group_seg = NR_LDPC_CUDA_GROUP_SEG;

/// TBs decoded together (one rate dematching per TB into its code blocks' slots of p_llr_dev, one decoder pass over
/// all the code blocks) must share these; at most NR_LDPC_CUDA_GROUP_SEG code blocks per group
bool nr_ldpc_cuda_same_group(const nrLDPC_TB_decoding_parameters_t *a, const nrLDPC_TB_decoding_parameters_t *b)
{
  return a->BG == b->BG && a->Z == b->Z && a->K == b->K && a->F == b->F && a->R == b->R
         && a->max_ldpc_iterations == b->max_ldpc_iterations && crcType(a->C, a->A) == crcType(b->C, b->A)
         && lenWithCrc(a->C, a->A) == lenWithCrc(b->C, b->A);
}

void nr_process_decode_group_cuda(nrLDPC_TB_decoding_parameters_t **tbs, int n)
{
  DevAssert(n > 0 && tbs[0] != NULL);
  nrLDPC_TB_decoding_parameters_t *t0 = tbs[0];
  const int Z = t0->Z;
  const int K = t0->K;
  const int Kprime = K - t0->F;
  t_nrLDPC_time_stats procTime = {0};

  pthread_mutex_lock(&decoder_mutex);
  // per TB: (copy to the device,) deinterleave, rate dematching into the HARQ buffer and the TB's code block slots
  int nseg = 0;
  for (int i = 0; i < n; i++) {
    nrLDPC_TB_decoding_parameters_t *segs = tbs[i];
    const int C = segs->C, E1 = segs->E, E2 = segs->E2, r_firstE2 = segs->first_rE2;
    AssertFatal(nseg + C <= NR_LDPC_CUDA_GROUP_SEG, "group of %d + %d code blocks\n", nseg, C);
    AssertFatal(segs->harq_unique_pid < d_array_size, "harq_unique_pid %d > %d\n", segs->harq_unique_pid, d_array_size);
    *segs->processedSegments = 0;
    const bool llr_dev = pageable || integrated || llr_on_device(segs->llr);
    if (!llr_dev)
      gpuMemcpyAsync_or_fail(harq_f_dev,
                             segs->llr,
                             ((r_firstE2 * E1) + (C - r_firstE2) * E2) * sizeof(int16_t),
                             gpuMemcpyHostToDevice,
                             decoderStreams[0]);
    launch_deinterleave_i16(segs->Qm, E1, E2, C, r_firstE2, harq_e_dev, llr_dev ? segs->llr : harq_f_dev, decoderStreams, 0);
    nr_rate_matching_ldpc_rx_cuda(segs->tbslbrm,
                                  segs->BG,
                                  Z,
                                  harq_d_array[segs->harq_unique_pid],
                                  harq_e_dev,
                                  p_llr_dev + (size_t)nseg * 68 * NR_LDPC_ZMAX,
                                  K,
                                  C,
                                  segs->rv_index,
                                  segs->d_to_be_cleared,
                                  E1,
                                  E2,
                                  r_firstE2,
                                  segs->F,
                                  Kprime - 2 * Z,
                                  decoderStreams,
                                  0);
    nseg += C;
  }
  // the 2Z punctured systematic LLRs of every code block, in one strided memset; the code blocks are 68 x ZMAX apart
  // in p_llr_dev (rate dematching output, decoder input), not 68 x Z
  gpuMemset2DAsync(p_llr_dev, 68 * NR_LDPC_ZMAX, 0, 2 * Z, nseg, decoderStreams[0]);

  start_meas(&t0->ts_ldpc_decode);
  t_nrLDPC_dec_params decParams = {.check_crc = check_crc};
  decParams.Z = Z;
  decParams.R = t0->R;
  decParams.BG = t0->BG;
  decParams.crc_type = crcType(t0->C, t0->A);
  decParams.Kprime = lenWithCrc(t0->C, t0->A);
  decParams.n_segments = nseg;
  decParams.outMode = nrLDPC_outMode_BIT;
  decParams.numMaxIter = t0->max_ldpc_iterations;
  uint8_t *out = n == 1 ? t0->c : group_out;
  LDPCdecoder_cuda(&decParams, p_llr_dev, out, &procTime, t0->abort_decode);
  stop_meas(&t0->ts_ldpc_decode);

  // per code block results, decoded bits to each TB
  const uint8_t *ok = ldpc_cuda_seg_ok();
  const int cb_bytes = K >> 3;
  nseg = 0;
  for (int i = 0; i < n; i++) {
    nrLDPC_TB_decoding_parameters_t *segs = tbs[i];
    if (n > 1)
      memcpy(segs->c, out + (size_t)nseg * cb_bytes, (size_t)segs->C * cb_bytes);
    uint32_t good = 0;
    for (int r = 0; r < (int)segs->C; r++) {
      segs->decodeSuccess[r] = ok[nseg + r];
      if (ok[nseg + r])
        good++;
      else
        memset(segs->c + (size_t)r * cb_bytes, 0, cb_bytes);
    }
    *segs->processedSegments = good;
    nseg += segs->C;
  }
  pthread_mutex_unlock(&decoder_mutex);
}

void nr_process_decode_segment_cuda(nrLDPC_TB_decoding_parameters_t *segs)
{
  nr_process_decode_group_cuda(&segs, 1);
}

#define MAXE 4 * 14 * 273 * 12 * 8 // 4 antennas, 14 symbols, 273 PRBs, 12 RE/prb, 8 bits/RE

void LDPCint_rm_init(int max_num_pxsch)
{
  group_out = gpuHostAlloc_or_fail(NR_LDPC_CUDA_GROUP_SEG * (22 * NR_LDPC_ZMAX / 8), gpuHostAllocDefault);
  LOG_I(NR_PHY, "RM init for %d pxsch\n", max_num_pxsch);
  LOG_I(NR_PHY, "Allocating device array for harq_d/harq_e \n");
  harq_e_dev = gpuMalloc_or_fail(MAXE * sizeof(int16_t));
  harq_d_array = malloc(sizeof(int16_t *) * max_num_pxsch);
  harq_d_array_dev = gpuMalloc_or_fail(sizeof(int16_t *) * max_num_pxsch);
  LOG_I(PHY, "Allocated %ld bytes for harq_d_array_dev @ %p\n", sizeof(int16_t *) * max_num_pxsch, harq_d_array_dev);
  for (int i = 0; i < max_num_pxsch; i++) {
    harq_d_array[i] = gpuMalloc_or_fail(MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * OAI_LDPC_DECODER_MAX_NUM_LLR * sizeof(int16_t));
    LOG_I(PHY,
          "Allocating %ld bytes for harq_d_array[%d] @ %p\n",
          MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * OAI_LDPC_DECODER_MAX_NUM_LLR * sizeof(int16_t),
          i,
          harq_d_array[i]);
  }
  gpuMemcpy(harq_d_array_dev, harq_d_array, sizeof(int16_t *) * max_num_pxsch, gpuMemcpyHostToDevice);
  if (!pageable && !integrated) {
    LOG_I(PHY, "Allocating device array for harq_f \n");
    harq_f_dev = gpuMalloc_or_fail(MAXE * sizeof(int16_t));
  }
  d_array_size = max_num_pxsch;
}

extern void LDPCinit_cuda(void);
extern void cuda_support_init();

int32_t nrLDPC_coding_init_cuda(int max_num_pxsch)
{
  cuda_support_init();

  LDPCinit_cuda();
  LDPCint_rm_init(max_num_pxsch);
  return 0;
}

void LDPCshutdown_cuda(void);

int32_t nrLDPC_coding_shutdown_cuda(void)
{
  LDPCshutdown_cuda();
  return 0;
}
