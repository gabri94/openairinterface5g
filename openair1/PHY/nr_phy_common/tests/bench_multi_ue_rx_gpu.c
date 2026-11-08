/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

// Multi-UE PDSCH receive chain on the GPU, slot by slot, with only the decoded TBs leaving the GPU:
//   1. DMRS channel estimation (bench kernel, all UEs in one launch)
//   2. fused 2-layer inner RX (nr_gpu_inner_rx_2layer): all UEs' data symbols in one launch, or one launch per UE (-P)
//   3. libldpc_cuda: deinterleaving, rate dematching into the HARQ buffer and decoding, the LLRs read in place
// The per-UE frequency-domain grids (what a GPU channel emulator would hand over) are generated once on the host:
// all-zero TBs (the transmitted bits are the scrambling sequence), per-PRB Rayleigh 2 x nb_rx channel, AWGN.
// 13 PDSCH symbols: one DMRS symbol (type 1, CDM group 0, no data on it) and 12 data symbols.
//
// usage: bench_multi_ue_rx_gpu [-u 1,4,12,24] [-p PRBs per UE (0: 273 / UEs)] [-q Qm] [-r code rate x1024]
//                              [-d mmse|lbest|ml] [-n nb_rx] [-s SNR dB] [-t slots] [-i LDPC iterations] [-a amplitude] [-P]
//                              [-C cpu list: LDPC on the CPU (libldpc.so, thread pool on these cores), LLRs copied to the host]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>
#include <getopt.h>

#include "PHY/gpu_compat.h"
#include "openair1/PHY/nr_phy_common/inc/nr_detector_gpu.h"
#include "openair1/PHY/CODING/nrLDPC_coding/nrLDPC_coding_interface.h"
#include "common/config/config_userapi.h"
#include "common/utils/LOG/log.h"
#include "bench_multi_ue_rx_gpu.h"
#include "common/utils/threadPool/thread-pool.h"

configmodule_interface_t *uniqCfg = NULL;
void exit_function(const char *file, const char *function, const int line, const char *s, const int assert)
{
  fprintf(stderr, "%s:%d %s: %s\n", file, line, function, s ? s : "");
  exit(1);
}

int nr_get_E(uint32_t G, uint8_t C, uint8_t Qm, uint8_t Nl, uint8_t r);
float nr_2layer_seed_lambda(uint32_t nvar, int shift);
int nr_2layer_log2_maxh(int avgs);
int nr_ml_llr_maxh_off(int mod_order, int lbest);
uint8_t log2_approx(uint32_t);
void crcTableInit(void);

#define CHECK(x)                                                    \
  do {                                                              \
    if ((x) != gpuSuccess) {                                        \
      fprintf(stderr, "%s:%d: %s failed\n", __FILE__, __LINE__, #x); \
      exit(1);                                                      \
    }                                                               \
  } while (0)

#define NSYM 12 // data symbols
static int cpu_ldpc; // -C
static double AMP = 400.0; // per-layer amplitude of a unit-power symbol through a unit channel (-a)

static double now_us(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec * 1e-3;
}

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint64_t rnd(void)
{
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return rng;
}
static double urand(void)
{
  return ((rnd() >> 11) + 0.5) / 9007199254740992.0;
}
static double grand(void)
{
  return sqrt(-2 * log(urand())) * cos(2 * M_PI * urand());
}
static int16_t sat16(double v)
{
  v = round(v);
  return v > 32767 ? 32767 : v < -32768 ? -32768 : (int16_t)v;
}

// 38.214 5.1.3.2, N_info > 3824
static uint32_t tbs_large(double ninfo, double R)
{
  const int n = (int)floor(log2(ninfo - 24)) - 5;
  const double np = fmax(3840, ldexp(1, n) * round((ninfo - 24) / ldexp(1, n)));
  if (R <= 0.25 || np > 8424) {
    const double C = ceil((np + 24) / (R <= 0.25 ? 3816 : 8424));
    return (uint32_t)(8 * C * ceil((np + 24) / (8 * C)) - 24);
  }
  return (uint32_t)(8 * ceil((np + 24) / 8) - 24);
}

// 38.211 5.1: one axis from the bits b0, b2, b4, b6 (I) or b1, b3, b5, b7 (Q)
static double qam_axis(const int *b, int qm)
{
  const double s0 = 1 - 2 * b[0];
  switch (qm) {
    case 2:
      return s0 / sqrt(2);
    case 4:
      return s0 * (2 - (1 - 2 * b[2])) / sqrt(10);
    case 6:
      return s0 * (4 - (1 - 2 * b[2]) * (2 - (1 - 2 * b[4]))) / sqrt(42);
    default:
      return s0 * (8 - (1 - 2 * b[2]) * (4 - (1 - 2 * b[4]) * (2 - (1 - 2 * b[6])))) / sqrt(170);
  }
}

typedef struct {
  uint32_t nprb, nre, stride, G, A;
  c16_t *d_rx; // [1 + NSYM][nb_rx][stride], symbol 0 = DMRS
  uint8_t *d_pilot;
  c16_t *d_ch;
  uint32_t *d_scr;
  int16_t *d_llr;
  int16_t *h_llr, *h_d; // CPU LDPC: LLRs on the host, HARQ buffer
  uint8_t *c; // decoded code blocks (host)
  uint32_t processed;
  decode_abort_t abort;
  nrLDPC_TB_decoding_parameters_t tb;
  nr_gpu_det_ws_t *ws; // per-UE launches
  nr_gpu_rx_sym_t sym[NSYM];
  int shift;
  float lambda;
} ue_t;

static void ue_setup(ue_t *u, int nprb, int nb_rx, int qm, double R, double snr_db, int det, int idx)
{
  memset(u, 0, sizeof(*u));
  u->nprb = nprb;
  u->nre = nprb * 12;
  u->stride = (u->nre + 15) & ~15u;
  u->G = NSYM * u->nre * qm * 2;
  u->A = tbs_large((double)NSYM * u->nre * R * qm * 2, R);
  const int BG = (u->A <= 292 || (u->A <= 3824 && R <= 0.67) || R <= 0.25) ? 2 : 1;
  nrLDPC_TB_decoding_parameters_t *tb = &u->tb;
  tb->harq_unique_pid = idx;
  tb->processedSegments = &u->processed;
  tb->nb_rb = nprb;
  tb->Qm = qm;
  tb->nb_layers = 2;
  tb->BG = BG;
  tb->rv_index = 0;
  tb->max_ldpc_iterations = 8;
  tb->abort_decode = &u->abort;
  init_abort(&u->abort);
  tb->G = u->G;
  tb->tbslbrm = 0; // full circular buffer
  tb->A = u->A;
  nr_segmentation(NULL, NULL, u->A + 24, &tb->C, &tb->K, &tb->Z, &tb->F, BG);
  tb->E = nr_get_E(tb->G, tb->C, qm, 2, 0);
  tb->E2 = tb->E;
  tb->first_rE2 = tb->C;
  int llrLen;
  tb->R = nr_get_R_ldpc_decoder(0, tb->E, BG, tb->Z, &llrLen, 0);
  for (uint32_t r = 1; r < tb->C; r++) {
    const int Er = nr_get_E(tb->G, tb->C, qm, 2, r);
    if (Er != tb->E) {
      tb->E2 = Er;
      tb->first_rE2 = r;
      tb->R2 = nr_get_R_ldpc_decoder(0, Er, BG, tb->Z, &llrLen, 0);
      break;
    }
  }
  u->c = calloc(tb->C, tb->K / 8 + 8);
  tb->c = u->c;

  // channel and noise
  const int np = u->nprb;
  double *H = malloc(sizeof(double) * np * nb_rx * 4); // [prb][rx][layer][re/im]
  for (int i = 0; i < np * nb_rx * 4; i++)
    H[i] = grand() * sqrt(0.5);
  const double nvar = 2 * AMP * AMP / pow(10, snr_db / 10), ns = sqrt(nvar / 2);
  // scrambling = transmitted bits
  const uint32_t nw = (u->G + 31) / 32;
  uint32_t *scr = malloc(nw * 4);
  for (uint32_t w = 0; w < nw; w++)
    scr[w] = (uint32_t)rnd();
  const size_t symsz = (size_t)nb_rx * u->stride;
  c16_t *rx = calloc((1 + NSYM) * symsz, sizeof(c16_t));
  uint8_t *pilot = malloc(u->nre / 2);
  for (uint32_t k = 0; k < u->nre / 2; k++)
    pilot[k] = rnd() & 3;
  for (uint32_t k = 0; k < u->nre; k++) {
    const int prb = k / 12;
    for (int a = 0; a < nb_rx; a++) {
      const double *h = &H[(prb * nb_rx + a) * 4];
      double yr = grand() * ns, yi = grand() * ns;
      if (!(k & 1)) {
        const uint8_t p = pilot[k / 2];
        const double pr = p & 1 ? -1 : 1, pi = p & 2 ? -1 : 1, w = (k / 2) & 1 ? -1 : 1;
        const double gr = h[0] + w * h[2], gi = h[1] + w * h[3];
        yr += AMP * (gr * pr - gi * pi);
        yi += AMP * (gr * pi + gi * pr);
      }
      rx[(size_t)a * u->stride + k] = (c16_t){sat16(yr), sat16(yi)};
    }
  }
  for (int s = 0; s < NSYM; s++)
    for (uint32_t k = 0; k < u->nre; k++) {
      double x[2][2];
      for (int l = 0; l < 2; l++) {
        int b[8];
        const size_t base = ((size_t)(s * u->nre + k) * 2 + l) * qm;
        for (int j = 0; j < qm; j++)
          b[j] = (scr[(base + j) / 32] >> ((base + j) % 32)) & 1;
        int bi[4], bq[4];
        for (int j = 0; j < qm / 2; j++) {
          bi[j] = b[2 * j];
          bq[j] = b[2 * j + 1];
        }
        // qam_axis indexes b[0], b[2], ...: spread the axis bits back to even positions
        int ei[8] = {0}, eq[8] = {0};
        for (int j = 0; j < qm / 2; j++) {
          ei[2 * j] = bi[j];
          eq[2 * j] = bq[j];
        }
        x[l][0] = qam_axis(ei, qm);
        x[l][1] = qam_axis(eq, qm);
      }
      const int prb = k / 12;
      for (int a = 0; a < nb_rx; a++) {
        const double *h = &H[(prb * nb_rx + a) * 4];
        double yr = grand() * ns, yi = grand() * ns;
        for (int l = 0; l < 2; l++) {
          yr += AMP * (h[2 * l] * x[l][0] - h[2 * l + 1] * x[l][1]);
          yi += AMP * (h[2 * l] * x[l][1] + h[2 * l + 1] * x[l][0]);
        }
        rx[(size_t)(1 + s) * symsz + (size_t)a * u->stride + k] = (c16_t){sat16(yr), sat16(yi)};
      }
    }
  // receiver scaling as nr_dlsch_demodulation sets it (channel level of the true channel)
  int avgs = 0;
  for (int l = 0; l < 2; l++)
    for (int a = 0; a < nb_rx; a++) {
      double e = 0;
      for (int p = 0; p < np; p++) {
        const double *h = &H[(p * nb_rx + a) * 4 + 2 * l];
        e += AMP * AMP * (h[0] * h[0] + h[1] * h[1]);
      }
      avgs = fmax(avgs, e / np);
    }
  u->shift = nr_2layer_log2_maxh(avgs) + nr_ml_llr_maxh_off(qm, det == NR_GPU_DET_LBEST) + log2_approx(nb_rx >> 1);
  u->lambda = nr_2layer_seed_lambda((uint32_t)nvar, u->shift);

  CHECK(gpuMalloc((void **)&u->d_rx, (1 + NSYM) * symsz * sizeof(c16_t)));
  CHECK(gpuMemcpy(u->d_rx, rx, (1 + NSYM) * symsz * sizeof(c16_t), gpuMemcpyHostToDevice));
  CHECK(gpuMalloc((void **)&u->d_pilot, u->nre / 2));
  CHECK(gpuMemcpy(u->d_pilot, pilot, u->nre / 2, gpuMemcpyHostToDevice));
  CHECK(gpuMalloc((void **)&u->d_ch, 2 * symsz * sizeof(c16_t)));
  CHECK(gpuMalloc((void **)&u->d_scr, nw * 4));
  CHECK(gpuMemcpy(u->d_scr, scr, nw * 4, gpuMemcpyHostToDevice));
  CHECK(gpuMalloc((void **)&u->d_llr, (size_t)u->G * sizeof(int16_t)));
  tb->llr = u->d_llr;
  if (cpu_ldpc) {
    u->h_llr = malloc((size_t)u->G * sizeof(int16_t));
    u->h_d = calloc((size_t)tb->C * 68 * tb->Z, sizeof(int16_t));
    tb->llr = u->h_llr;
    tb->d = u->h_d;
  }
  for (int s = 0; s < NSYM; s++)
    u->sym[s] = (nr_gpu_rx_sym_t){.rxF = u->d_rx + (1 + s) * symsz,
                                  .chF = u->d_ch,
                                  .re_map = NULL,
                                  .scramble = u->d_scr,
                                  .scramble_offset = (uint32_t)s * u->nre * 2 * qm,
                                  .llr_cw = u->d_llr + (size_t)s * u->nre * 2 * qm,
                                  .length = u->nre,
                                  .stride = u->stride,
                                  .ch_stride = u->stride,
                                  .cpe = {32767, 0}};
  free(H);
  free(scr);
  free(rx);
  free(pilot);
}

static void ue_free(ue_t *u)
{
  gpuFree(u->d_rx);
  gpuFree(u->d_pilot);
  gpuFree(u->d_ch);
  gpuFree(u->d_scr);
  gpuFree(u->d_llr);
  if (u->ws)
    nr_gpu_det_ws_free(u->ws);
  free(u->c);
  free(u->h_llr);
  free(u->h_d);
}

static int cmp_d(const void *a, const void *b)
{
  const double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

typedef struct {
  double mean, p99, max;
} st_t;
static st_t stats(double *v, int n)
{
  double s = 0;
  for (int i = 0; i < n; i++)
    s += v[i];
  qsort(v, n, sizeof(*v), cmp_d);
  return (st_t){s / n, v[(int)(0.99 * (n - 1))], v[n - 1]};
}

int main(int argc, char **argv)
{
  if ((uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY)) == 0) {
    fprintf(stderr, "configuration module init failed\n");
    exit(1);
  }
  logInit();
  set_glog(OAILOG_WARNING);
  crcTableInit(); // check_crc of libldpc_cuda.so uses the tables of this executable

  char ues_s[256] = "1,4,12,24";
  char cpus[512] = "";
  int nprb_arg = 0, qm = 6, rate = 616, det = NR_GPU_DET_MMSE, nb_rx = 4, slots = 200, iters = 8, per_ue = 0;
  double snr = 30;
  int c;
  while ((c = getopt(argc, argv, "u:p:q:r:d:n:s:t:i:a:PC:")) != -1) {
    switch (c) {
      case 'u':
        snprintf(ues_s, sizeof(ues_s), "%s", optarg);
        break;
      case 'p':
        nprb_arg = atoi(optarg);
        break;
      case 'q':
        qm = atoi(optarg);
        break;
      case 'r':
        rate = atoi(optarg);
        break;
      case 'd':
        det = !strcmp(optarg, "ml") ? NR_GPU_DET_ML : !strcmp(optarg, "lbest") ? NR_GPU_DET_LBEST : NR_GPU_DET_MMSE;
        break;
      case 'n':
        nb_rx = atoi(optarg);
        break;
      case 's':
        snr = atof(optarg);
        break;
      case 't':
        slots = atoi(optarg);
        break;
      case 'i':
        iters = atoi(optarg);
        break;
      case 'a':
        AMP = atof(optarg);
        break;
      case 'C':
        snprintf(cpus, sizeof(cpus), "%s", optarg);
        cpu_ldpc = 1;
        break;
      case 'P':
        per_ue = 1;
        break;
      default:
        fprintf(stderr, "see the header of %s\n", __FILE__);
        exit(1);
    }
  }
  int ue_list[64], n_list = 0, max_ue = 0;
  for (char *t = strtok(ues_s, ","); t && n_list < 64; t = strtok(NULL, ","))
    if ((ue_list[n_list] = atoi(t)) > 0) {
      max_ue = ue_list[n_list] > max_ue ? ue_list[n_list] : max_ue;
      n_list++;
    }

  tpool_t pool;
  if (cpu_ldpc)
    initTpool(cpus, &pool, false);
  void *lib = dlopen(cpu_ldpc ? "libldpc.so" : "libldpc_cuda.so", RTLD_NOW);
  if (!lib) {
    fprintf(stderr, "%s\n", dlerror());
    exit(1);
  }
  nrLDPC_coding_init_t *ldpc_init = dlsym(lib, "nrLDPC_coding_init");
  nrLDPC_coding_decoder_t *ldpc_decode = dlsym(lib, "nrLDPC_coding_decoder");
  if (!ldpc_init || !ldpc_decode || ldpc_init(max_ue) != 0) {
    fprintf(stderr, "LDPC library init failed\n");
    exit(1);
  }

  gpuDeviceProp_t prop;
  CHECK(gpuGetDeviceProperties(&prop, 0));
  static const char *dn[] = {"MMSE", "L-best", "", "ML"};
  printf("%s (%d SMs): 2 layers, %d RX, Qm %d, rate %d/1024, %s, SNR %.1f dB, amplitude %.0f, %d data symbols, %d slots, inner RX %s, LDPC %s%s\n",
         prop.name, prop.multiProcessorCount, nb_rx, qm, rate, dn[det], snr, AMP, NSYM, slots, per_ue ? "per UE" : "batched", cpu_ldpc ? "CPU on " : "GPU", cpus);
  printf("%4s %5s %7s %3s %4s | %8s %8s | %8s %8s | %8s %8s %8s | %7s %8s %6s\n", "UEs", "PRB", "TBS", "C", "Zc", "chest", "", "innerRX",
         "", "LDPC", "", "", "total", "", "TB ok");
  printf("%4s %5s %7s %3s %4s | %8s %8s | %8s %8s | %8s %8s %8s | %7s %8s %6s\n", "", "/UE", "", "", "", "mean", "max", "mean", "max", "mean", "p99", "max",
         "mean", "p99/max", "");

  gpuStream_t st;
  CHECK(gpuStreamCreateWithFlags(&st, gpuStreamNonBlocking));
  gpuEvent_t ev[3];
  for (int i = 0; i < 3; i++)
    CHECK(gpuEventCreate(&ev[i]));

  for (int li = 0; li < n_list; li++) {
    const int N = ue_list[li];
    const int nprb = nprb_arg > 0 ? nprb_arg : 273 / N;
    ue_t *ue = calloc(N, sizeof(ue_t));
    for (int i = 0; i < N; i++) {
      ue_setup(&ue[i], nprb, nb_rx, qm, rate / 1024.0, snr, det, i);
      ue[i].tb.max_ldpc_iterations = iters;
    }
    if (ue[0].tb.BG != 1 || ue[0].tb.Z < 128) {
      printf("%4d %5d %7u: BG %d Zc %u, not on the GPU decoder (BG 1, Zc >= 128 only)\n", N, nprb, ue[0].A, ue[0].tb.BG, ue[0].tb.Z);
      for (int i = 0; i < N; i++)
        ue_free(&ue[i]);
      free(ue);
      continue;
    }
    // chest descriptors (device), noise accumulators
    mue_chest_t *hc = calloc(N, sizeof(*hc)), *dc;
    float *d_nvar;
    CHECK(gpuMalloc((void **)&d_nvar, N * sizeof(float)));
    for (int i = 0; i < N; i++)
      hc[i] = (mue_chest_t){ue[i].d_rx, ue[i].d_pilot, ue[i].d_ch, d_nvar + i, ue[i].nprb, ue[i].stride};
    CHECK(gpuMalloc((void **)&dc, N * sizeof(*dc)));
    CHECK(gpuMemcpy(dc, hc, N * sizeof(*dc), gpuMemcpyHostToDevice));
    // inner RX: one batch (shared shift / lambda: the UEs' channel statistics are the same) or one ws per UE
    nr_gpu_rx_sym_t *all = malloc(sizeof(*all) * N * NSYM);
    for (int i = 0; i < N; i++)
      memcpy(&all[i * NSYM], ue[i].sym, sizeof(ue[i].sym));
    nr_gpu_det_ws_t *ws = NULL;
    int shift_diff = 0;
    for (int i = 0; i < N; i++)
      shift_diff += ue[i].shift != ue[0].shift;
    if (per_ue) {
      for (int i = 0; i < N; i++)
        ue[i].ws = nr_gpu_det_ws_alloc(2 * NSYM * ue[i].nre);
    } else {
      ws = nr_gpu_det_ws_alloc_batch(2 * N * NSYM * ue[0].nre, N * NSYM);
      if (!ws) {
        fprintf(stderr, "ws alloc failed\n");
        exit(1);
      }
    }
    nrLDPC_TB_decoding_parameters_t *tbs = malloc(sizeof(*tbs) * N);
    nrLDPC_slot_decoding_parameters_t slot = {.nb_TBs = N, .TBs = tbs, .threadPool = cpu_ldpc ? &pool : NULL};
    double d2h = 0;

    const int warm = 10, T = slots + warm;
    double *t_ch = malloc(sizeof(double) * T), *t_rx = malloc(sizeof(double) * T), *t_ld = malloc(sizeof(double) * T),
           *t_tot = malloc(sizeof(double) * T);
    long ok = 0, tot = 0;
    double llr_neg = 0, llr_abs = 0, nvar_est = 0;
    for (int it = 0; it < T; it++) {
      const double t0 = now_us();
      CHECK(gpuMemsetAsync(d_nvar, 0, N * sizeof(float), st));
      CHECK(gpuEventRecord(ev[0], st));
      if (mue_chest(dc, N, nb_rx, nprb, st))
        exit(1);
      CHECK(gpuEventRecord(ev[1], st));
      if (per_ue) {
        for (int i = 0; i < N; i++)
          if (nr_gpu_inner_rx_2layer(ue[i].sym, NSYM, nb_rx, qm, det, ue[i].lambda, ue[i].shift, ue[i].ws, st))
            exit(1);
      } else if (nr_gpu_inner_rx_2layer(all, N * NSYM, nb_rx, qm, det, ue[0].lambda, ue[0].shift, ws, st)) {
        fprintf(stderr, "inner RX launch failed\n");
        exit(1);
      }
      CHECK(gpuEventRecord(ev[2], st));
      CHECK(gpuStreamSynchronize(st));
      double c_d2h = 0;
      if (cpu_ldpc) { // a CPU UE has its LLRs on the host: the copy is counted neither as LDPC nor in the total
        const double c0 = now_us();
        for (int i = 0; i < N; i++)
          CHECK(gpuMemcpy(ue[i].h_llr, ue[i].d_llr, (size_t)ue[i].G * sizeof(int16_t), gpuMemcpyDeviceToHost));
        c_d2h = now_us() - c0;
        if (it >= warm)
          d2h += c_d2h;
      }
      const double t1 = now_us();
      for (int i = 0; i < N; i++) {
        ue[i].processed = 0;
        ue[i].tb.d_to_be_cleared = true;
        tbs[i] = ue[i].tb;
      }
      ldpc_decode(&slot);
      const double t2 = now_us();

      float a, b;
      CHECK(gpuEventElapsedTime(&a, ev[0], ev[1]));
      CHECK(gpuEventElapsedTime(&b, ev[1], ev[2]));
      if (it >= warm) {
        const int k = it - warm;
        t_ch[k] = a * 1e3;
        t_rx[k] = b * 1e3;
        t_ld[k] = t2 - t1;
        t_tot[k] = t2 - t0 - c_d2h;
        for (int i = 0; i < N; i++)
          ok += ue[i].processed == ue[i].tb.C;
        tot += N;
      }
      if (it == 0) {
        const uint32_t G = ue[0].G;
        int16_t *l = malloc(G * sizeof(int16_t));
        CHECK(gpuMemcpy(l, ue[0].d_llr, G * sizeof(int16_t), gpuMemcpyDeviceToHost));
        for (uint32_t j = 0; j < G; j++) {
          llr_neg += l[j] < 0;
          llr_abs += abs(l[j]);
        }
        llr_neg /= G;
        llr_abs /= G;
        free(l);
        float nv[1];
        CHECK(gpuMemcpy(nv, d_nvar, sizeof(float), gpuMemcpyDeviceToHost));
        nvar_est = nv[0] / (ue[0].nprb * nb_rx);
      }
    }
    const st_t sc = stats(t_ch, slots), sr = stats(t_rx, slots), sl = stats(t_ld, slots), stt = stats(t_tot, slots);
    printf("%4d %5d %7u %3u %4u | %8.1f %8.1f | %8.1f %8.1f | %8.1f %8.1f %8.1f | %7.1f %4.0f/%-4.0f %5.1f%%\n", N, nprb, ue[0].A, ue[0].tb.C,
           ue[0].tb.Z, sc.mean, sc.max, sr.mean, sr.max, sl.mean, sl.p99, sl.max, stt.mean, stt.p99, stt.max, 100.0 * ok / tot);
    printf("      UE0: shift %d lambda %.1f (UEs with another shift: %d), LLR mean |.| %.1f, negative %.2f%%, noise est/true %.2f, LLR copy to host %.1f us/slot (not counted)\n",
           ue[0].shift, ue[0].lambda, shift_diff, llr_abs, 100 * llr_neg, nvar_est / (2 * AMP * AMP / pow(10, snr / 10)), d2h / slots);
    fflush(stdout);
    for (int i = 0; i < N; i++)
      ue_free(&ue[i]);
    if (ws)
      nr_gpu_det_ws_free(ws);
    gpuFree(dc);
    gpuFree(d_nvar);
    free(hc);
    free(all);
    free(tbs);
    free(ue);
    free(t_ch);
    free(t_rx);
    free(t_ld);
    free(t_tot);
  }
  return 0;
}
