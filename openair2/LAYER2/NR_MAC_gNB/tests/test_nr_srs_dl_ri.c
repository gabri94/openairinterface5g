/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* Unit tests for nr_srs_dl_ri.c (SRS-reciprocity DL rank + subspace xcorr).
 * Plain C, no framework: builds synthetic channels, quantizes them the way
 * cuPHY reports the normalized channel IQ matrix, and checks the estimates. */

#include "../nr_srs_dl_ri.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int n_fail = 0;
static int n_pass = 0;

#define CHECK(cond, ...)                     \
  do {                                       \
    if (cond) {                              \
      n_pass++;                              \
    } else {                                 \
      n_fail++;                              \
      printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      printf(__VA_ARGS__);                   \
      printf("\n");                          \
    }                                        \
  } while (0)

/* deterministic PRNG (xorshift32) + Box-Muller-ish uniform-sum gaussian */
static uint32_t rng_state = 0x1234abcd;
static double frand(void) /* uniform [-1, 1) */
{
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return (int32_t)rng_state / 2147483648.0;
}
static double grand(void) /* ~N(0,1), sum of 12 uniforms */
{
  double s = 0;
  for (int i = 0; i < 12; i++)
    s += frand() * 0.5 + 0.5;
  return s - 6.0;
}

#define MAX_NG 64
#define MAX_NU 4
#define MAX_NPRG 68

typedef struct {
  double re[MAX_NPRG][MAX_NG][MAX_NU];
  double im[MAX_NPRG][MAX_NG][MAX_NU];
} chan_t;

/* quantize to the SRS.IND layout: element (u,g,p) at (u*ng+g)*nprg+p */
static void quantize(const chan_t *h, int ng, int nu, int nprg, int iq_bits, double scale, void *out)
{
  for (int u = 0; u < nu; u++)
    for (int g = 0; g < ng; g++)
      for (int p = 0; p < nprg; p++) {
        const int idx = (u * ng + g) * nprg + p;
        const double re = h->re[p][g][u] * scale;
        const double im = h->im[p][g][u] * scale;
        if (iq_bits == 8) {
          int8_t *m = (int8_t *)out;
          m[2 * idx] = (int8_t)lrint(fmax(-127, fmin(127, re)));
          m[2 * idx + 1] = (int8_t)lrint(fmax(-127, fmin(127, im)));
        } else {
          int16_t *m = (int16_t *)out;
          m[2 * idx] = (int16_t)lrint(fmax(-32767, fmin(32767, re)));
          m[2 * idx + 1] = (int16_t)lrint(fmax(-32767, fmin(32767, im)));
        }
      }
}

/* h += amp * a b^T where a (ng), b (nu) are given complex vectors; a gets a
 * per-PRG frequency-selective phase ramp so PRGs differ like a real channel */
static void add_rank1_mode(chan_t *h,
                           int ng,
                           int nu,
                           int nprg,
                           const double *a_re,
                           const double *a_im,
                           const double *b_re,
                           const double *b_im,
                           double amp,
                           double delay)
{
  for (int p = 0; p < nprg; p++) {
    const double ph = 2.0 * M_PI * delay * p / nprg;
    const double c = cos(ph), s = sin(ph);
    for (int g = 0; g < ng; g++) {
      const double ar = a_re[g] * c - a_im[g] * s;
      const double ai = a_re[g] * s + a_im[g] * c;
      for (int u = 0; u < nu; u++) {
        h->re[p][g][u] += amp * (ar * b_re[u] - ai * b_im[u]);
        h->im[p][g][u] += amp * (ar * b_im[u] + ai * b_re[u]);
      }
    }
  }
}

static void random_unit_vec(int n, double *vre, double *vim)
{
  double norm = 0;
  for (int i = 0; i < n; i++) {
    vre[i] = grand();
    vim[i] = grand();
    norm += vre[i] * vre[i] + vim[i] * vim[i];
  }
  norm = 1.0 / sqrt(norm);
  for (int i = 0; i < n; i++) {
    vre[i] *= norm;
    vim[i] *= norm;
  }
}

/* Gram-Schmidt step: v -= (u^H v) u, assuming u is unit norm */
static void project_out(int n, double *vre, double *vim, const double *ure, const double *uim)
{
  double dre = 0, dim = 0;
  for (int i = 0; i < n; i++) {
    dre += ure[i] * vre[i] + uim[i] * vim[i];
    dim += ure[i] * vim[i] - uim[i] * vre[i];
  }
  for (int i = 0; i < n; i++) {
    vre[i] -= dre * ure[i] - dim * uim[i];
    vim[i] -= dre * uim[i] + dim * ure[i];
  }
  double norm = 0;
  for (int i = 0; i < n; i++)
    norm += vre[i] * vre[i] + vim[i] * vim[i];
  norm = 1.0 / sqrt(norm);
  for (int i = 0; i < n; i++) {
    vre[i] *= norm;
    vim[i] *= norm;
  }
}

static void add_noise(chan_t *h, int ng, int nu, int nprg, double sigma)
{
  for (int p = 0; p < nprg; p++)
    for (int g = 0; g < ng; g++)
      for (int u = 0; u < nu; u++) {
        h->re[p][g][u] += sigma * grand();
        h->im[p][g][u] += sigma * grand();
      }
}

/* build a channel with `modes` mutually-orthogonal spatial modes and return
 * the quantized matrix in buf */
static void make_channel(chan_t *h,
                         int ng,
                         int nu,
                         int nprg,
                         int modes,
                         const double *mode_amp,
                         double noise_sigma)
{
  static double are[MAX_NU][MAX_NG], aim[MAX_NU][MAX_NG];
  static double bre[MAX_NU][MAX_NU], bim[MAX_NU][MAX_NU];
  memset(h, 0, sizeof(*h));
  for (int m = 0; m < modes; m++) {
    random_unit_vec(ng, are[m], aim[m]);
    random_unit_vec(nu, bre[m], bim[m]);
    for (int k = 0; k < m; k++) {
      project_out(ng, are[m], aim[m], are[k], aim[k]);
      project_out(nu, bre[m], bim[m], bre[k], bim[k]);
    }
    add_rank1_mode(h, ng, nu, nprg, are[m], aim[m], bre[m], bim[m], mode_amp[m], 0.5 + m);
  }
  if (noise_sigma > 0)
    add_noise(h, ng, nu, nprg, noise_sigma);
}

/* ---------------- eigensolver tests ---------------- */

static void test_eig_known_2x2(void)
{
  /* [[2, 1], [1, 2]] -> {3, 1} */
  nr_srs_cd_t g[4] = {{2, 0}, {1, 0}, {1, 0}, {2, 0}};
  double ev[2];
  nr_srs_cd_t v[4];
  nr_srs_hermitian_eig(2, g, ev, v);
  CHECK(fabs(ev[0] - 3.0) < 1e-12 && fabs(ev[1] - 1.0) < 1e-12, "2x2 real eig: got %f %f", ev[0], ev[1]);

  /* [[2, i], [-i, 2]] -> {3, 1} */
  nr_srs_cd_t gc[4] = {{2, 0}, {0, 1}, {0, -1}, {2, 0}};
  nr_srs_hermitian_eig(2, gc, ev, v);
  CHECK(fabs(ev[0] - 3.0) < 1e-12 && fabs(ev[1] - 1.0) < 1e-12, "2x2 complex eig: got %f %f", ev[0], ev[1]);
}

static void test_eig_random_4x4(void)
{
  for (int trial = 0; trial < 50; trial++) {
    /* random Hermitian G = B^H B */
    nr_srs_cd_t b[16], g[16];
    for (int i = 0; i < 16; i++)
      b[i] = (nr_srs_cd_t){grand(), grand()};
    double trace = 0;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) {
        nr_srs_cd_t acc = {0, 0};
        for (int k = 0; k < 4; k++) {
          nr_srs_cd_t t = {b[k * 4 + i].re, -b[k * 4 + i].im};
          acc.re += t.re * b[k * 4 + j].re - t.im * b[k * 4 + j].im;
          acc.im += t.re * b[k * 4 + j].im + t.im * b[k * 4 + j].re;
        }
        g[i * 4 + j] = acc;
        if (i == j)
          trace += acc.re;
      }

    double ev[4];
    nr_srs_cd_t v[16];
    nr_srs_hermitian_eig(4, g, ev, v);

    /* eigenvalues descending, non-negative, sum = trace */
    CHECK(ev[0] >= ev[1] && ev[1] >= ev[2] && ev[2] >= ev[3], "eig order trial %d", trial);
    CHECK(ev[3] > -1e-9 * trace, "PSD eigenvalue trial %d: %g", trial, ev[3]);
    CHECK(fabs(ev[0] + ev[1] + ev[2] + ev[3] - trace) < 1e-9 * (trace + 1), "trace trial %d", trial);

    /* residual ||G v - ev v|| per eigenpair */
    for (int m = 0; m < 4; m++) {
      double res = 0, vnorm = 0;
      for (int i = 0; i < 4; i++) {
        nr_srs_cd_t acc = {0, 0};
        for (int k = 0; k < 4; k++) {
          acc.re += g[i * 4 + k].re * v[k * 4 + m].re - g[i * 4 + k].im * v[k * 4 + m].im;
          acc.im += g[i * 4 + k].re * v[k * 4 + m].im + g[i * 4 + k].im * v[k * 4 + m].re;
        }
        acc.re -= ev[m] * v[i * 4 + m].re;
        acc.im -= ev[m] * v[i * 4 + m].im;
        res += acc.re * acc.re + acc.im * acc.im;
        vnorm += v[i * 4 + m].re * v[i * 4 + m].re + v[i * 4 + m].im * v[i * 4 + m].im;
      }
      CHECK(sqrt(res) < 1e-8 * (trace + 1), "eigenpair residual trial %d mode %d: %g", trial, m, sqrt(res));
      CHECK(fabs(vnorm - 1.0) < 1e-9, "eigenvector norm trial %d mode %d: %g", trial, m, vnorm);
    }
  }
}

/* ---------------- rank estimation tests ---------------- */

static chan_t H; /* too big for the stack */
static int16_t iq16[2 * MAX_NU * MAX_NG * MAX_NPRG];
static int8_t iq8[2 * MAX_NU * MAX_NG * MAX_NPRG];
static nr_srs_cf_t basisA[MAX_NPRG * MAX_NU * MAX_NG];
static nr_srs_cf_t basisB[MAX_NPRG * MAX_NU * MAX_NG];

static void test_rank_synthetic(void)
{
  const int ngs[] = {4, 32, 64};
  for (int gi = 0; gi < 3; gi++) {
    const int ng = ngs[gi];
    const int nprg = 68;

    /* rank-1, nu=2 and nu=4 */
    for (int nu = 2; nu <= 4; nu += 2) {
      double amp[1] = {1.0};
      make_channel(&H, ng, nu, nprg, 1, amp, 0.0);
      quantize(&H, ng, nu, nprg, 16, 4000.0, iq16);
      int r = nr_srs_dl_rank_estimate(iq16, 16, ng, nu, nprg, 2, NR_SRS_DL_RI_REL_THRESH, 4, 1, NULL, NULL);
      CHECK(r == 1, "rank-1 ng=%d nu=%d: got %d", ng, nu, r);
    }

    /* rank-2 equal power, nu=2 */
    {
      double amp[2] = {1.0, 1.0};
      make_channel(&H, ng, 2, nprg, 2, amp, 0.0);
      quantize(&H, ng, 2, nprg, 16, 4000.0, iq16);
      int r = nr_srs_dl_rank_estimate(iq16, 16, ng, 2, nprg, 2, NR_SRS_DL_RI_REL_THRESH, 4, 1, NULL, NULL);
      CHECK(r == 2, "rank-2 ng=%d nu=2: got %d", ng, r);
    }

    /* rank-4 equal power, nu=4; also max_rank cap */
    {
      double amp[4] = {1.0, 1.0, 1.0, 1.0};
      make_channel(&H, ng, 4, nprg, 4, amp, 0.0);
      quantize(&H, ng, 4, nprg, 16, 4000.0, iq16);
      int r = nr_srs_dl_rank_estimate(iq16, 16, ng, 4, nprg, 2, NR_SRS_DL_RI_REL_THRESH, 4, 1, NULL, NULL);
      CHECK(r == 4, "rank-4 ng=%d: got %d", ng, r);
      r = nr_srs_dl_rank_estimate(iq16, 16, ng, 4, nprg, 2, NR_SRS_DL_RI_REL_THRESH, 2, 1, NULL, NULL);
      CHECK(r == 2, "rank-4 capped to 2, ng=%d: got %d", ng, r);
    }
  }
}

static void test_rank_threshold(void)
{
  const int ng = 32, nu = 2, nprg = 68;

  /* second mode at -26 dB power (amplitude 0.05): below the -13 dB gate -> 1 */
  double weak[2] = {1.0, 0.05};
  make_channel(&H, ng, nu, nprg, 2, weak, 0.0);
  quantize(&H, ng, nu, nprg, 16, 4000.0, iq16);
  int r = nr_srs_dl_rank_estimate(iq16, 16, ng, nu, nprg, 2, NR_SRS_DL_RI_REL_THRESH, 4, 1, NULL, NULL);
  CHECK(r == 1, "weak 2nd mode (-26 dB): got %d", r);

  /* second mode at -6 dB power (amplitude 0.5): above the gate -> 2 */
  double strong[2] = {1.0, 0.5};
  make_channel(&H, ng, nu, nprg, 2, strong, 0.0);
  quantize(&H, ng, nu, nprg, 16, 4000.0, iq16);
  r = nr_srs_dl_rank_estimate(iq16, 16, ng, nu, nprg, 2, NR_SRS_DL_RI_REL_THRESH, 4, 1, NULL, NULL);
  CHECK(r == 2, "-6 dB 2nd mode: got %d", r);
}

static void test_rank_noise_and_quant(void)
{
  const int ng = 32, nu = 2, nprg = 68;

  /* rank-1 with noise at ~20 dB element SNR: per-element |h| ~ 1/sqrt(ng*nu)
   * on unit modes, so sigma = 0.1/sqrt(ng*nu) */
  double amp[1] = {1.0};
  make_channel(&H, ng, nu, nprg, 1, amp, 0.1 / sqrt((double)ng * nu));
  quantize(&H, ng, nu, nprg, 16, 4000.0, iq16);
  int r = nr_srs_dl_rank_estimate(iq16, 16, ng, nu, nprg, 2, NR_SRS_DL_RI_REL_THRESH, 4, 1, NULL, NULL);
  CHECK(r == 1, "rank-1 @20dB SNR: got %d", r);

  /* same channel through the int8 representation (coarser) */
  quantize(&H, ng, nu, nprg, 8, 100.0, iq8);
  r = nr_srs_dl_rank_estimate(iq8, 8, ng, nu, nprg, 2, NR_SRS_DL_RI_REL_THRESH, 4, 1, NULL, NULL);
  CHECK(r == 1, "rank-1 int8: got %d", r);

  /* rank-2 with noise, int8 */
  double amp2[2] = {1.0, 0.8};
  make_channel(&H, ng, nu, nprg, 2, amp2, 0.05 / sqrt((double)ng * nu));
  quantize(&H, ng, nu, nprg, 8, 100.0, iq8);
  r = nr_srs_dl_rank_estimate(iq8, 8, ng, nu, nprg, 2, NR_SRS_DL_RI_REL_THRESH, 4, 1, NULL, NULL);
  CHECK(r == 2, "rank-2 int8 noisy: got %d", r);
}

static void test_rank_degenerate(void)
{
  const int ng = 32, nu = 2, nprg = 68;
  memset(iq16, 0, sizeof(iq16));
  int r = nr_srs_dl_rank_estimate(iq16, 16, ng, nu, nprg, 2, NR_SRS_DL_RI_REL_THRESH, 4, 1, NULL, NULL);
  CHECK(r == 0, "all-zero chest: got %d", r);

  /* nu=1 is always rank 1 (if nonzero) */
  double amp[1] = {1.0};
  make_channel(&H, ng, 1, nprg, 1, amp, 0.0);
  quantize(&H, ng, 1, nprg, 16, 4000.0, iq16);
  r = nr_srs_dl_rank_estimate(iq16, 16, ng, 1, nprg, 2, NR_SRS_DL_RI_REL_THRESH, 4, 1, NULL, NULL);
  CHECK(r == 1, "nu=1: got %d", r);

  /* invalid args */
  CHECK(nr_srs_dl_rank_estimate(NULL, 16, ng, nu, nprg, 2, 0.05f, 4, 1, NULL, NULL) == 0, "NULL iq");
  CHECK(nr_srs_dl_rank_estimate(iq16, 12, ng, nu, nprg, 2, 0.05f, 4, 1, NULL, NULL) == 0, "bad width");
  CHECK(nr_srs_dl_rank_estimate(iq16, 16, ng, 5, nprg, 2, 0.05f, 4, 1, NULL, NULL) == 0, "nu>4");
}

static void test_decimation_consistency(void)
{
  const int ng = 32, nu = 2, nprg = 68;
  double amp[2] = {1.0, 0.7};
  make_channel(&H, ng, nu, nprg, 2, amp, 0.02 / sqrt((double)ng * nu));
  quantize(&H, ng, nu, nprg, 16, 4000.0, iq16);
  int r1 = nr_srs_dl_rank_estimate(iq16, 16, ng, nu, nprg, 1, NR_SRS_DL_RI_REL_THRESH, 4, 1, NULL, NULL);
  int r4 = nr_srs_dl_rank_estimate(iq16, 16, ng, nu, nprg, 4, NR_SRS_DL_RI_REL_THRESH, 4, 1, NULL, NULL);
  CHECK(r1 == 2 && r4 == 2, "decimation consistency: step1=%d step4=%d", r1, r4);
}

/* ---------------- basis + xcorr tests ---------------- */

static void test_basis_orthonormal(void)
{
  const int ng = 32, nu = 4, nprg = 68;
  double amp[4] = {1.0, 0.9, 0.8, 0.7};
  make_channel(&H, ng, nu, nprg, 4, amp, 0.0);
  quantize(&H, ng, nu, nprg, 16, 4000.0, iq16);
  int nb = 0;
  int r = nr_srs_dl_rank_estimate(iq16, 16, ng, nu, nprg, 2, NR_SRS_DL_RI_REL_THRESH, 4, 1, basisA, &nb);
  CHECK(r == 4, "basis rank: got %d", r);
  CHECK(nb == 34, "basis nprg: got %d", nb);

  /* per PRG: U^H U = I */
  double max_err = 0;
  for (int p = 0; p < nb; p++)
    for (int i = 0; i < r; i++)
      for (int j = 0; j < r; j++) {
        const nr_srs_cf_t *ui = &basisA[(p * r + i) * ng];
        const nr_srs_cf_t *uj = &basisA[(p * r + j) * ng];
        double dre = 0, dim = 0;
        for (int g = 0; g < ng; g++) {
          dre += (double)ui[g].re * uj[g].re + (double)ui[g].im * uj[g].im;
          dim += (double)ui[g].re * uj[g].im - (double)ui[g].im * uj[g].re;
        }
        const double target = (i == j) ? 1.0 : 0.0;
        const double err = fabs(sqrt(dre * dre + dim * dim) - target);
        if (err > max_err)
          max_err = err;
      }
  CHECK(max_err < 1e-3, "basis orthonormality: max err %g", max_err); /* quantization-limited */
}

static void test_xcorr(void)
{
  const int ng = 16, nu = 2, nprg = 68;

  /* same channel twice -> xcorr ~ 1 */
  double amp[2] = {1.0, 0.9};
  make_channel(&H, ng, nu, nprg, 2, amp, 0.0);
  quantize(&H, ng, nu, nprg, 16, 4000.0, iq16);
  int nbA = 0, nbB = 0;
  int ra = nr_srs_dl_rank_estimate(iq16, 16, ng, nu, nprg, 2, NR_SRS_DL_RI_REL_THRESH, 4, 1, basisA, &nbA);
  int rb = nr_srs_dl_rank_estimate(iq16, 16, ng, nu, nprg, 2, NR_SRS_DL_RI_REL_THRESH, 4, 1, basisB, &nbB);
  float xc = nr_srs_subspace_xcorr(basisA, ra, basisB, rb, ng, nbA < nbB ? nbA : nbB);
  CHECK(xc > 0.99f && xc <= 1.001f, "self xcorr: got %f", xc);

  /* orthogonal column spaces: UE A lives on antennas 0-7, UE B on 8-15 */
  static double are[2][MAX_NG], aim[2][MAX_NG], bre[2][MAX_NU], bim[2][MAX_NU];
  memset(&H, 0, sizeof(H));
  for (int m = 0; m < 2; m++) {
    memset(are[m], 0, sizeof(are[m]));
    memset(aim[m], 0, sizeof(aim[m]));
    random_unit_vec(8, are[m], aim[m]); /* support on antennas 0-7 only */
    random_unit_vec(nu, bre[m], bim[m]);
    if (m == 1) {
      project_out(8, are[1], aim[1], are[0], aim[0]);
      project_out(nu, bre[1], bim[1], bre[0], bim[0]);
    }
    add_rank1_mode(&H, ng, nu, nprg, are[m], aim[m], bre[m], bim[m], 1.0, 0.5 + m);
  }
  quantize(&H, ng, nu, nprg, 16, 4000.0, iq16);
  ra = nr_srs_dl_rank_estimate(iq16, 16, ng, nu, nprg, 2, NR_SRS_DL_RI_REL_THRESH, 4, 1, basisA, &nbA);

  memset(&H, 0, sizeof(H));
  for (int m = 0; m < 2; m++) {
    memset(are[m], 0, sizeof(are[m]));
    memset(aim[m], 0, sizeof(aim[m]));
    double tmp_re[8], tmp_im[8];
    random_unit_vec(8, tmp_re, tmp_im);
    memcpy(&are[m][8], tmp_re, sizeof(tmp_re)); /* support on antennas 8-15 only */
    memcpy(&aim[m][8], tmp_im, sizeof(tmp_im));
    random_unit_vec(nu, bre[m], bim[m]);
    if (m == 1) {
      project_out(ng, are[1], aim[1], are[0], aim[0]);
      project_out(nu, bre[1], bim[1], bre[0], bim[0]);
    }
    add_rank1_mode(&H, ng, nu, nprg, are[m], aim[m], bre[m], bim[m], 1.0, 0.5 + m);
  }
  quantize(&H, ng, nu, nprg, 16, 4000.0, iq16);
  rb = nr_srs_dl_rank_estimate(iq16, 16, ng, nu, nprg, 2, NR_SRS_DL_RI_REL_THRESH, 4, 1, basisB, &nbB);

  xc = nr_srs_subspace_xcorr(basisA, ra, basisB, rb, ng, nbA < nbB ? nbA : nbB);
  CHECK(xc >= 0.0f && xc < 0.01f, "orthogonal xcorr: got %f (ra=%d rb=%d)", xc, ra, rb);

  /* invalid input */
  CHECK(nr_srs_subspace_xcorr(NULL, 1, basisB, 1, ng, 1) < 0, "xcorr NULL");
}

int main(void)
{
  test_eig_known_2x2();
  test_eig_random_4x4();
  test_rank_synthetic();
  test_rank_threshold();
  test_rank_noise_and_quant();
  test_rank_degenerate();
  test_decimation_consistency();
  test_basis_orthonormal();
  test_xcorr();

  printf("%d passed, %d failed\n", n_pass, n_fail);
  return n_fail ? 1 : 0;
}
