/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

// DMRS channel estimation of bench_multi_ue_rx_gpu: one thread per (UE, PRB, RX antenna)

#include "PHY/gpu_compat.h"
#include "bench_multi_ue_rx_gpu.h"

static __device__ inline int16_t sat16(float v)
{
  v = rintf(v);
  return (int16_t)fminf(32767.f, fmaxf(-32768.f, v));
}

__global__ void k_mue_chest(const mue_chest_t *u, int nb_rx)
{
  const mue_chest_t &c = u[blockIdx.y];
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t prb = t / nb_rx, a = t % nb_rx;
  if (prb >= c.nprb)
    return;
  const c16_t *y = c.rx_dmrs + (size_t)a * c.stride + prb * 12;
  const uint8_t *p = c.pilot + prb * 6;
  float h[3][2][2]; // pair, port, re/im
  float m[2][2] = {{0, 0}, {0, 0}};
  for (int j = 0; j < 3; j++) {
    float l[2][2];
    for (int k = 0; k < 2; k++) {
      const c16_t v = y[4 * j + 2 * k];
      const uint8_t pi = p[2 * j + k];
      const float sr = pi & 1 ? -1.f : 1.f, si = pi & 2 ? -1.f : 1.f;
      // y conj(pilot) / |pilot|^2, pilot = sr + j si
      l[k][0] = 0.5f * (v.r * sr + v.i * si);
      l[k][1] = 0.5f * (v.i * sr - v.r * si);
    }
    for (int q = 0; q < 2; q++) {
      h[j][0][q] = 0.5f * (l[0][q] + l[1][q]);
      h[j][1][q] = 0.5f * (l[0][q] - l[1][q]);
      m[0][q] += h[j][0][q] / 3;
      m[1][q] += h[j][1][q] / 3;
    }
  }
  // per port, the 3 pair estimates have variance nvar / 4 each: E[sum |h_j - mean|^2] = nvar / 2 per port
  float e = 0;
  for (int j = 0; j < 3; j++)
    for (int l = 0; l < 2; l++)
      for (int q = 0; q < 2; q++)
        e += (h[j][l][q] - m[l][q]) * (h[j][l][q] - m[l][q]);
  atomicAdd(c.nvar, e);
  for (int l = 0; l < 2; l++) {
    const c16_t v = {sat16(m[l][0]), sat16(m[l][1])};
    c16_t *o = c.chF + (size_t)(l * nb_rx + a) * c.stride + prb * 12;
    for (int i = 0; i < 12; i++)
      o[i] = v;
  }
}

extern "C" int mue_chest(const mue_chest_t *u, int n_ue, int nb_rx, uint32_t max_prb, void *stream)
{
  const uint32_t n = max_prb * nb_rx;
  const dim3 grid((n + 127) / 128, n_ue);
  k_mue_chest<<<grid, 128, 0, (gpuStream_t)stream>>>(u, nb_rx);
  return gpuGetLastError() == gpuSuccess ? 0 : -1;
}
