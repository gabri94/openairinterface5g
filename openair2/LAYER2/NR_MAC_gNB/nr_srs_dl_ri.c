/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*! \file nr_srs_dl_ri.c
 * \brief SRS-reciprocity DL rank estimation and multi-UE subspace correlation.
 *
 * Approach: per PRG build the nu x nu Hermitian Gram matrix G = H^H H
 * (accumulated over the ng gNB antennas), eigendecompose it, and count the
 * eigenvalues within rel_thresh of the strongest. The per-PRG ranks are
 * combined by majority vote. The Gram eigenvectors V and eigenvalues
 * sigma^2 also give an orthonormal basis of the channel column space,
 * U = H V Sigma^-1 (the left singular vectors), used for the cross-UE
 * orthogonality metric.
 *
 * All internal accumulation is double: with ng = 64 antennas of int16 IQ the
 * Gram entries reach ~7e13, past int32 and float's 24-bit mantissa.
 */

#include "nr_srs_dl_ri.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- tiny complex-double helpers (row-major n x n, n <= 4) ---- */

static inline nr_srs_cd_t cd_add(nr_srs_cd_t a, nr_srs_cd_t b)
{
  return (nr_srs_cd_t){a.re + b.re, a.im + b.im};
}

static inline nr_srs_cd_t cd_mul(nr_srs_cd_t a, nr_srs_cd_t b)
{
  return (nr_srs_cd_t){a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

static inline nr_srs_cd_t cd_conj(nr_srs_cd_t a)
{
  return (nr_srs_cd_t){a.re, -a.im};
}

static inline nr_srs_cd_t cd_scale(nr_srs_cd_t a, double s)
{
  return (nr_srs_cd_t){a.re * s, a.im * s};
}

static inline double cd_abs2(nr_srs_cd_t a)
{
  return a.re * a.re + a.im * a.im;
}

/* Stable complex Jacobi rotation for M = [[a, b], [conj(b), d]], a, d real.
 * Factor the phase out (P = diag(1, e^-i.phi) makes the off-diagonal real
 * |b|), then use the classic small-angle rotation: tan(2.theta) picked via
 * t = sign(tau)/(|tau| + sqrt(1 + tau^2)), which avoids the cancellation of
 * the "exact eigenvector" construction when |b| is small. U = P*J annihilates
 * the off-diagonal of U^H M U to machine precision. */
static void herm2x2_diagonalizer(double a, nr_srs_cd_t b, double d, nr_srs_cd_t u[2][2])
{
  const double absb = sqrt(cd_abs2(b));
  if (absb <= 0.0) { /* already diagonal */
    u[0][0] = (nr_srs_cd_t){1, 0};
    u[0][1] = (nr_srs_cd_t){0, 0};
    u[1][0] = (nr_srs_cd_t){0, 0};
    u[1][1] = (nr_srs_cd_t){1, 0};
    return;
  }
  const double tau = (d - a) / (2.0 * absb);
  const double t = (tau >= 0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau * tau));
  const double c = 1.0 / sqrt(1.0 + t * t);
  const double s = t * c;
  const nr_srs_cd_t em = cd_scale(cd_conj(b), 1.0 / absb); /* e^-i.phi */
  u[0][0] = (nr_srs_cd_t){c, 0};
  u[0][1] = (nr_srs_cd_t){s, 0};
  u[1][0] = cd_scale(em, -s);
  u[1][1] = cd_scale(em, c);
}

void nr_srs_hermitian_eig(int n, const nr_srs_cd_t *g, double *ev, nr_srs_cd_t *v)
{
  nr_srs_cd_t a[NR_SRS_DL_RI_MAX_PORTS][NR_SRS_DL_RI_MAX_PORTS];
  nr_srs_cd_t vv[NR_SRS_DL_RI_MAX_PORTS][NR_SRS_DL_RI_MAX_PORTS];

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      a[i][j] = g[i * n + j];
      vv[i][j] = (nr_srs_cd_t){i == j ? 1.0 : 0.0, 0.0};
    }

  /* Scale-aware tolerance from the diagonal (Gram => trace = sum |h|^2).
   * 1e-10 relative off-diagonal is far below what the -13 dB rank threshold
   * and the float basis output can resolve; machine-precision convergence
   * would roughly double the sweep count for nothing. */
  double scale = 0.0;
  for (int i = 0; i < n; i++)
    scale += fabs(a[i][i].re);
  const double tol2 = (scale * 1e-10) * (scale * 1e-10) + 1e-300;

  for (int sweep = 0; sweep < 16; sweep++) {
    double off2 = 0.0;
    for (int p = 0; p < n - 1; p++)
      for (int q = p + 1; q < n; q++)
        off2 += cd_abs2(a[p][q]);
    if (off2 <= tol2 * n * n)
      break;

    for (int p = 0; p < n - 1; p++) {
      for (int q = p + 1; q < n; q++) {
        if (cd_abs2(a[p][q]) <= tol2)
          continue;
        nr_srs_cd_t u[2][2];
        herm2x2_diagonalizer(a[p][p].re, a[p][q], a[q][q].re, u);

        /* A <- U^H A U applied on rows/cols p, q; V <- V U */
        for (int k = 0; k < n; k++) { /* column update: A(:,p), A(:,q) */
          nr_srs_cd_t akp = a[k][p], akq = a[k][q];
          a[k][p] = cd_add(cd_mul(akp, u[0][0]), cd_mul(akq, u[1][0]));
          a[k][q] = cd_add(cd_mul(akp, u[0][1]), cd_mul(akq, u[1][1]));
        }
        for (int k = 0; k < n; k++) { /* row update: A(p,:), A(q,:) */
          nr_srs_cd_t apk = a[p][k], aqk = a[q][k];
          a[p][k] = cd_add(cd_mul(cd_conj(u[0][0]), apk), cd_mul(cd_conj(u[1][0]), aqk));
          a[q][k] = cd_add(cd_mul(cd_conj(u[0][1]), apk), cd_mul(cd_conj(u[1][1]), aqk));
        }
        if (v) /* eigenvector accumulation only when requested */
          for (int k = 0; k < n; k++) {
            nr_srs_cd_t vkp = vv[k][p], vkq = vv[k][q];
            vv[k][p] = cd_add(cd_mul(vkp, u[0][0]), cd_mul(vkq, u[1][0]));
            vv[k][q] = cd_add(cd_mul(vkp, u[0][1]), cd_mul(vkq, u[1][1]));
          }
        /* enforce exact Hermitian structure on the rotated pair */
        a[p][q] = (nr_srs_cd_t){0, 0};
        a[q][p] = (nr_srs_cd_t){0, 0};
        a[p][p].im = 0;
        a[q][q].im = 0;
      }
    }
  }

  /* sort eigen-pairs descending */
  int order[NR_SRS_DL_RI_MAX_PORTS];
  for (int i = 0; i < n; i++)
    order[i] = i;
  for (int i = 0; i < n - 1; i++)
    for (int j = i + 1; j < n; j++)
      if (a[order[j]][order[j]].re > a[order[i]][order[i]].re) {
        int t = order[i];
        order[i] = order[j];
        order[j] = t;
      }
  for (int i = 0; i < n; i++) {
    ev[i] = a[order[i]][order[i]].re;
    if (v)
      for (int k = 0; k < n; k++)
        v[k * n + i] = vv[k][order[i]];
  }
}

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
                            int *basis_nprg)
{
  if (basis_nprg)
    *basis_nprg = 0;
  if (!iq || (iq_bits != 8 && iq_bits != 16) || ng < 1 || nu < 1 || nu > NR_SRS_DL_RI_MAX_PORTS || nprg < 1)
    return 0;
  if (prg_step < 1)
    prg_step = 1;
  if (basis_step < 1)
    basis_step = 1;
  if (max_rank < 1)
    max_rank = 1;
  if (max_rank > nu)
    max_rank = nu;

  /* The SRS.IND layout ((u*ng + g)*nprg + p, PRG innermost) makes per-PRG
   * access stride nprg elements: a naive per-PRG Gram re-reads the whole
   * matrix nu^2/2 times with ~KB strides (measured ~10x slower). Instead do
   * ONE layout-friendly pass transposing the decimated PRGs into contiguous
   * scratch h[pdec][g][u], and cache the per-PRG eigendecomposition so the
   * basis pass doesn't recompute it. */
  const int nprg_dec = (nprg + prg_step - 1) / prg_step;
  nr_srs_cf_t *h = NULL; /* [nprg_dec][ng][nu] */
  double *ev_all = NULL; /* [nprg_dec][nu] */
  nr_srs_cd_t *v_all = NULL; /* [nprg_dec][nu*nu], only when basis wanted */
  h = (nr_srs_cf_t *)malloc((size_t)nprg_dec * ng * nu * sizeof(*h));
  ev_all = (double *)malloc((size_t)nprg_dec * nu * sizeof(*ev_all));
  if (basis)
    v_all = (nr_srs_cd_t *)malloc((size_t)nprg_dec * nu * nu * sizeof(*v_all));
  if (!h || !ev_all || (basis && !v_all)) {
    free(h);
    free(ev_all);
    free(v_all);
    return 0;
  }

  /* transpose/convert: source rows (fixed u,g) are contiguous in p */
  for (int u = 0; u < nu; u++)
    for (int g = 0; g < ng; g++) {
      nr_srs_cf_t *dst = &h[(size_t)g * nu + u]; /* += nprg-dec stride ng*nu */
      if (iq_bits == 8) {
        const int8_t *src = (const int8_t *)iq + 2 * ((size_t)(u * ng + g) * nprg);
        for (int pd = 0; pd < nprg_dec; pd++, dst += (size_t)ng * nu) {
          dst->re = src[2 * pd * prg_step];
          dst->im = src[2 * pd * prg_step + 1];
        }
      } else {
        const int16_t *src = (const int16_t *)iq + 2 * ((size_t)(u * ng + g) * nprg);
        for (int pd = 0; pd < nprg_dec; pd++, dst += (size_t)ng * nu) {
          dst->re = src[2 * pd * prg_step];
          dst->im = src[2 * pd * prg_step + 1];
        }
      }
    }

  /* guards against an all-zero / collapsed chest: an eigenvalue below this is
   * "no energy" no matter the quantization (min nonzero |h|^2 = 1) */
  const double abs_floor = 0.5;

  /* per-PRG Gram + eigendecomposition, majority vote over per-PRG ranks */
  int histo[NR_SRS_DL_RI_MAX_PORTS + 1] = {0};
  for (int pd = 0; pd < nprg_dec; pd++) {
    const nr_srs_cf_t *hp = &h[(size_t)pd * ng * nu];
    nr_srs_cd_t gm[NR_SRS_DL_RI_MAX_PORTS * NR_SRS_DL_RI_MAX_PORTS];
    for (int u1 = 0; u1 < nu; u1++)
      for (int u2 = u1; u2 < nu; u2++) {
        double are = 0, aim = 0;
        for (int g = 0; g < ng; g++) { /* conj(h[g][u1]) * h[g][u2] */
          const nr_srs_cf_t h1 = hp[g * nu + u1];
          const nr_srs_cf_t h2 = hp[g * nu + u2];
          are += (double)h1.re * h2.re + (double)h1.im * h2.im;
          aim += (double)h1.re * h2.im - (double)h1.im * h2.re;
        }
        gm[u1 * nu + u2] = (nr_srs_cd_t){are, aim};
        gm[u2 * nu + u1] = (nr_srs_cd_t){are, -aim};
      }

    double *ev = &ev_all[(size_t)pd * nu];
    nr_srs_hermitian_eig(nu, gm, ev, v_all ? &v_all[(size_t)pd * nu * nu] : NULL);

    if (ev[0] <= abs_floor) {
      histo[0]++;
      continue;
    }
    int r = 1;
    for (int i = 1; i < nu; i++)
      if (ev[i] > (double)rel_thresh * ev[0])
        r++;
    if (r > max_rank)
      r = max_rank;
    histo[r]++;
  }

  int rank = 0, best = -1;
  for (int r = 0; r <= NR_SRS_DL_RI_MAX_PORTS; r++)
    if (histo[r] > best) { /* ties resolved towards the lower rank */
      best = histo[r];
      rank = r;
    }

  /* optional: orthonormal column-space basis U = H V Sigma^-1, `rank` modes
   * per basis PRG (further decimated by basis_step) */
  if (rank > 0 && basis) {
    int pb = 0;
    for (int pd = 0; pd < nprg_dec; pd += basis_step, pb++) {
      const nr_srs_cf_t *hp = &h[(size_t)pd * ng * nu];
      const double *ev = &ev_all[(size_t)pd * nu];
      const nr_srs_cd_t *v = &v_all[(size_t)pd * nu * nu];
      for (int m = 0; m < rank; m++) {
        nr_srs_cf_t *col = &basis[((size_t)pb * rank + m) * ng];
        if (ev[m] <= abs_floor) { /* deficient PRG: emit a zero mode, keep layout */
          memset(col, 0, ng * sizeof(*col));
          continue;
        }
        const double inv_sigma = 1.0 / sqrt(ev[m]);
        for (int g = 0; g < ng; g++) {
          double are = 0, aim = 0;
          for (int u = 0; u < nu; u++) {
            const nr_srs_cf_t hg = hp[g * nu + u];
            const nr_srs_cd_t vm = v[u * nu + m];
            are += (double)hg.re * vm.re - (double)hg.im * vm.im;
            aim += (double)hg.re * vm.im + (double)hg.im * vm.re;
          }
          col[g] = (nr_srs_cf_t){(float)(are * inv_sigma), (float)(aim * inv_sigma)};
        }
      }
    }
    if (basis_nprg)
      *basis_nprg = pb;
  }

  free(h);
  free(ev_all);
  free(v_all);
  return rank;
}

float nr_srs_subspace_xcorr(const nr_srs_cf_t *basis_a,
                            int ra,
                            const nr_srs_cf_t *basis_b,
                            int rb,
                            int ng,
                            int nprg)
{
  if (!basis_a || !basis_b || ra < 1 || rb < 1 || ng < 1 || nprg < 1)
    return -1.0f;

  double acc = 0.0;
  for (int p = 0; p < nprg; p++) {
    for (int i = 0; i < ra; i++) {
      const nr_srs_cf_t *ua = &basis_a[(p * ra + i) * ng];
      for (int j = 0; j < rb; j++) {
        const nr_srs_cf_t *ub = &basis_b[(p * rb + j) * ng];
        double dre = 0.0, dim = 0.0;
        for (int g = 0; g < ng; g++) { /* conj(ua) . ub */
          dre += (double)ua[g].re * ub[g].re + (double)ua[g].im * ub[g].im;
          dim += (double)ua[g].re * ub[g].im - (double)ua[g].im * ub[g].re;
        }
        acc += dre * dre + dim * dim;
      }
    }
  }
  const int rmin = ra < rb ? ra : rb;
  return (float)(acc / ((double)nprg * rmin));
}
