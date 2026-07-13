/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*! \file nr_srs_dl_ri.h
 * \brief SRS-reciprocity DL rank estimation and multi-UE subspace correlation.
 *
 * Pure math over the SRS channel-estimate matrix delivered in SRS.indication
 * (normalized channel IQ matrix, SCF 222 Table 3-133). No OAI dependencies so
 * the module is unit-testable standalone; callers in the MAC do the locking,
 * config lookups and buffer management.
 *
 * Channel matrix layout (as unpacked by unpack_nr_srs_normalized_channel_iq_matrix):
 *   element (uI, gI, pI) at index uI*ng*nprg + gI*nprg + pI  (PRG innermost),
 *   int8 or int16 interleaved I,Q per element.
 */

#ifndef NR_SRS_DL_RI_H
#define NR_SRS_DL_RI_H

#include <stdint.h>

#define NR_SRS_DL_RI_MAX_PORTS 4

/* PRG decimation step for the rank/basis computation. The decision is
 * wideband (the dynamic-BFW CVI request is full-band, num_prgs=1), so
 * skipping PRGs only cheapens the vote: 273 PRGs / 4 still gives a
 * 68-sample majority. */
#define NR_SRS_DL_RI_PRG_STEP 4

/* A spatial mode counts towards the rank when its Gram eigenvalue is within
 * this factor of the strongest one: 0.05 = -13 dB per-layer power ratio. */
#define NR_SRS_DL_RI_REL_THRESH 0.05f

/* Additional decimation (in units of already-decimated PRGs) for the stored
 * orthogonality basis: the cross-UE metric doesn't need full frequency
 * resolution and its cost is quadratic in the number of UEs. */
#define NR_SRS_DL_RI_BASIS_STEP 4

typedef struct {
  float re;
  float im;
} nr_srs_cf_t;

typedef struct {
  double re;
  double im;
} nr_srs_cd_t;

/* Eigendecomposition of an n x n Hermitian matrix, n <= 4 (complex Jacobi).
 * g: row-major n*n input (only read); ev: eigenvalues, descending;
 * v: optional row-major n*n unitary, column m = eigenvector of ev[m]. */
void nr_srs_hermitian_eig(int n, const nr_srs_cd_t *g, double *ev, nr_srs_cd_t *v);

/* Estimate the wideband rank of the SRS channel estimate.
 *
 * iq:        channel matrix in the layout above
 * iq_bits:   8 or 16 (normalized_iq_representation 0 / 1)
 * ng:        gNB antenna elements, nu: UE SRS ports (nu <= 4), nprg: PRGs
 * prg_step:  PRG decimation for the rank vote (>= 1)
 * rel_thresh: per-PRG eigenvalue ratio for a mode to count (e.g. 0.05)
 * max_rank:  cap (maxMIMO layers etc.), >= 1
 * basis_step: extra decimation for the stored basis, in units of
 *            already-decimated PRGs (>= 1); effective basis grid is
 *            every prg_step*basis_step-th PRG
 * basis:     optional; on rank r > 0 receives, per basis PRG, an
 *            orthonormal basis of the channel column space: element
 *            (pb, mode, g) at index (pb*r + mode)*ng + g. Caller must
 *            provide capacity for
 *            ceil(ceil(nprg/prg_step)/basis_step) * min(nu, max_rank) * ng
 *            elements.
 * basis_nprg: optional; receives the number of basis PRGs written.
 *
 * Returns the majority-vote rank in 1..min(nu, max_rank), or 0 if the input
 * is degenerate (zero/garbage chest) or on internal allocation failure. */
int nr_srs_dl_rank_estimate(const void *iq,
                            int iq_bits,
                            int ng,
                            int nu,
                            int nprg,
                            int prg_step,
                            float rel_thresh,
                            int max_rank,
                            int basis_step,
                            nr_srs_cf_t *basis,
                            int *basis_nprg);

/* Mean squared subspace overlap between two UEs' channel column spaces:
 *   (1 / (nprg * min(ra, rb))) * sum_prg ||Ua^H Ub||_F^2   in [0, 1]
 * 0 = orthogonal channels (ideal MU-MIMO pair), 1 = fully overlapping.
 * Both bases must be over the same PRG grid and antenna count; pass
 * nprg = min of the two stored decimated-PRG counts. */
float nr_srs_subspace_xcorr(const nr_srs_cf_t *basis_a,
                            int ra,
                            const nr_srs_cf_t *basis_b,
                            int rb,
                            int ng,
                            int nprg);

#endif /* NR_SRS_DL_RI_H */
