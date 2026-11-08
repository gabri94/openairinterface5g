/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef BENCH_MULTI_UE_RX_GPU_H
#define BENCH_MULTI_UE_RX_GPU_H

#include <stdint.h>
#include "platform_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// One UE's DMRS symbol for mue_chest (device pointers): DMRS type 1, CDM group 0 (even subcarriers), ports 1000 /
/// 1001 separated by the frequency OCC over the pairs (4n, 4n + 2), one DMRS symbol, no data in CDM group 1.
typedef struct {
  const c16_t *rx_dmrs; ///< received DMRS symbol [nb_rx][stride]
  const uint8_t *pilot; ///< pilot of each DMRS RE (6 per PRB): bit 0 real sign, bit 1 imaginary sign (1: negative)
  c16_t *chF; ///< channel estimate [2 layers][nb_rx][stride], per PRB (flat over its 12 REs)
  float *nvar; ///< += noise variance seen by each (PRB, antenna) (OCC pair spread around the PRB mean)
  uint32_t nprb;
  uint32_t stride;
} mue_chest_t;

/// LS + frequency OCC despreading + PRB average for n_ue UEs in one launch (u: device array)
int mue_chest(const mue_chest_t *u, int n_ue, int nb_rx, uint32_t max_prb, void *stream);

#ifdef __cplusplus
}
#endif

#endif
