# SRS-reciprocity dynamic downlink beamforming (Aerial)

This document describes the gNB **L2 (NR MAC)** support for SRS-reciprocity-based
dynamic downlink beamforming with the NVIDIA Aerial L1. The DL beamforming
weights are not taken from a fixed codebook: they are derived per UE from the
uplink SRS channel estimate and re-applied on the downlink (TDD reciprocity).

The feature is **Aerial-only** (`ENABLE_AERIAL`): OAI runs as the FAPI VNF
(L2) and NVIDIA Aerial / cuPHY runs as the PNF (L1). L2 never sees the raw
channel estimate or computes weights itself — it tells L1 *which* cached SRS
estimate to use and *when*, and L1 computes and applies the weights. This keeps
the heavy linear algebra and the fronthaul beamforming entirely in cuPHY.

> Scope: this document focuses on the **L2/MAC** design and how to enable it.
> The L1 (Aerial/cuPHY) and the O-RU are only summarized — see the Aerial and
> O-RAN documentation for those layers.

---

## 1. High-level flow

```
        UE                 L2 (OAI NR MAC)              L1 (Aerial/cuPHY)             O-RU
        │                       │                            │                         │
        │  ── SRS (periodic) ──────────────────────────────────────────────────────► │
        │                       │                            │ ◄── UL SRS U-plane ──── │
        │                       │  ① SRS PDU (handle=buf)    │
        │                       │ ─────────────────────────► │ compute SRS chest,
        │                       │                            │ store in chest buf[buf]
        │                       │  ② SRS.indication          │
        │                       │ ◄───────────────────────── │ (+ normalized chest)
        │            mark buf READY                          │
        │                       │  ③ DL_BFW_CVI.request      │
        │            (per slot, │ ─────────────────────────► │ compute BFW from
        │             1 ahead)  │     handle=buf, digBF=0    │ chest buf[buf];
        │                       │                            │ store in BFW coeff ring
        │                       │  ④ DL_TTI.request (PDSCH)  │
        │                       │ ─────────────────────────► │ apply BFW → SE11 weights
        │                       │                            │ ──── C-plane SE11 BFW ─► │ beamform
        │ ◄──────────────────────────────────────────────────────── beamformed PDSCH ─ │
```

1. The UE sounds **periodic SRS**. When L2 schedules that SRS, it allocates a
   per-UE *chest buffer* and stamps its index into the SRS PDU `handle`, telling
   cuPHY where to cache this UE's channel estimate.
2. cuPHY estimates the channel from the SRS and returns an `SRS.indication`; L2
   marks the buffer **READY**.
3. On every DL slot, L2's emitter issues a **BFW_CVI request** for UEs with a
   READY chest, referencing the buffer `handle`, *one slot ahead* of the PDSCH
   that will use it.
4. cuPHY computes the beamforming weights from the cached estimate and applies
   them to the PDSCH; the weights travel to the O-RU as O-RAN C-plane Section
   Extension 11 (SE11), and the RU beamforms.

---

## 2. L2 (NR MAC) design

Everything below lives under `openair2/LAYER2/NR_MAC_gNB/` unless noted.

### 2.1 Enabling the mode

A new analog-beamforming mode `SRS_DYNAMIC_BFW` (enum value 3) is selected from
the gNB configuration:

```
gNBs.[0].set_analog_beamforming = "dynamic_bfw";
```

`MACRLC_nr_paramdef.h` maps the string set `{none, preconfigured, lophy,
dynamic_bfw}` to `{NO_BEAM_MODE, PRECONFIGURED_BEAM_IDX, LOPHY_BEAM_IDX,
SRS_DYNAMIC_BFW}`. The mode is stored in `gNB_MAC_INST.beam_info.beam_mode` and
gates all the logic below. Because the weights come from SRS, the UE **must**
sound SRS (`do_SRS = "periodic"`).

### 2.2 The SRS chest buffer pool

L2 does not hold channel estimates; it tracks *which* cuPHY buffer holds each
UE's estimate. `nr_mac_gNB.h` defines a small per-cell pool:

```c
#define NR_SRS_CHEST_BUF_POOL_SIZE 64        // <= SCF 222.10.04 NUM_SRS_CHEST_BUFFERS max (1023)

typedef enum {
  SRS_CHEST_BUF_FREE = 0,
  SRS_CHEST_BUF_ALLOCATED,  // index handed to an SRS PDU, chest not yet stored
  SRS_CHEST_BUF_READY,      // cuPHY stored a fresh chest; BFW request pending
  SRS_CHEST_BUF_REQUESTED,  // BFW_CVI request issued / sounding in flight
} nr_srs_chest_buf_state_t;

typedef struct { nr_srs_chest_buf_state_t state; uint16_t rnti; uint8_t ng; uint8_t nu; } nr_srs_chest_buf_t;
```

`gNB_MAC_INST.srs_chest_buf[NR_SRS_CHEST_BUF_POOL_SIZE]` holds the pool. The pool
API (in `gNB_scheduler_srs.c`, declared in `mac_proto.h`):

| function | purpose |
|---|---|
| `nr_srs_chest_buf_get` | allocate (or reuse) a buffer for an RNTI → ALLOCATED |
| `nr_srs_chest_buf_mark_ready` | on SRS.ind: mark READY, record `ng`/`nu` dims |
| `nr_srs_chest_buf_get_ready` | peek a READY buffer's dims (non-consuming) |
| `nr_srs_chest_buf_peek_ready` | is there a READY buffer for this RNTI? |
| `nr_srs_chest_buf_consume_ready` | READY → REQUESTED (one-shot consume) |
| `nr_srs_chest_buf_free_ue` | release on UE removal (`mac_remove_nr_ue`) |

### 2.3 SRS → L1: stamping the buffer handle

In `nr_configure_srs()` (`gNB_scheduler_srs.c`), when `beam_mode ==
SRS_DYNAMIC_BFW`, L2 allocates a buffer and encodes its index into the SRS PDU
`handle` (bits 8..23), which cuPHY uses to decide where to cache the estimate:

```c
int buf = nr_srs_chest_buf_get(nrmac, UE->rnti);
if (buf >= 0) {
  srs_pdu->handle = ((uint32_t)(buf & 0xFFFF)) << 8;   // bits 8..23 = chest buffer index
  nrmac->srs_chest_buf[buf].state = SRS_CHEST_BUF_REQUESTED;  // sounding in flight
}
```

The buffer is marked `REQUESTED` for the sounding window so the per-slot emitter
and the consumption gate pause for that UE until the estimate actually lands —
otherwise BFW requests issued during the window are rejected by cuPHY with
`0x40 SRS_CHEST_BUFF_BAD_STATE`.

### 2.4 L1 → SRS.indication: marking READY

`handle_nr_srs_measurements()` (`gNB_scheduler_ulsch.c`) runs on the
`SRS.indication`. For `SRS_DYNAMIC_BFW` it marks the UE's buffer READY and records
the estimate dimensions (`num_gnb_antenna_elements` = Ng, `num_ue_srs_ports` =
Nu):

```c
if (nrmac->beam_info.beam_mode == SRS_DYNAMIC_BFW)
  nr_srs_chest_buf_mark_ready(nrmac, srs_ind->rnti,
                              iq_matrix.num_gnb_antenna_elements,
                              iq_matrix.num_ue_srs_ports);
```

The same handler also emits the normalized channel matrix to the T-Tracer
(`T_GNB_MAC_SRS_CHANNEL_MATRIX`) for offline inspection.

### 2.5 The per-slot BFW_CVI emitter

`nr_sched_dynamic_bfw()` (`gNB_scheduler_dlsch.c`) is called every slot from
`gNB_dlsch_ulsch_scheduler()` (`gNB_scheduler.c`), after DL scheduling:

```c
#ifdef ENABLE_AERIAL
  // emit BFW_CVI every slot (proactive) so cuPHY's slot-indexed BFW coeff ring
  // stays full; PDSCH(N) reads ring[(N-1)%4] and finds fresh coeffs.
  nr_sched_dynamic_bfw(RC.nrmac[module_idP], frame, slot);
#endif
```

It is **proactive**: it emits a request one slot ahead of PDSCH consumption so
cuPHY's slot-indexed BFW coefficient ring is always populated. It is
**non-consuming** (`get_ready`, READY-only) so the same READY estimate is
re-stamped each slot until a fresh sounding replaces it. The request carries the
buffer `handle` and sets `dig_bf_interfaces = 0`, which tells cuPHY to use the
SRS-derived weights rather than a codebook PMI.

### 2.6 The BFW_CVI FAPI message (vendor extension)

A vendor P7 message carries the request (`nfapi_nr_interface_scf.h`):

- `NFAPI_NR_PHY_MSG_TYPE_DL_BFW_CVI_REQUEST = 0x90` (and UL `0x91`).
- `nfapi_nr_bfw_cvi_request_t` → `{sfn, slot, num_groups, group_list[]}`; each
  group → `{rb_start, rb_size, num_prgs, prg_size, ue_list[]}`; each UE entry →
  `{rnti, handle, pdu_idx, gnb_ant_idx_start/end, ue_ant_idx[]}`
  (`handle` bits 8..23 = the SRS chest buffer index).
- Limits: `NFAPI_NR_MAX_BFW_CVI_GROUPS=16`, `…_UES_PER_GROUP=12`, `…_UE_ANTS=4`.

Pack/unpack lives in `nfapi/open-nFAPI/fapi/src/nr_fapi_p7.c`; the nvIPC
transport and the `oai_fapi_dl_bfw_cvi_req()` send helper are in
`nfapi/oai_integration/aerial/fapi_nvIPC.c` and `fapi_vnf_p7.c`.

A separate config-time TLV sizes cuPHY's chest pool: `NUM_SRS_CHEST_BUFFERS`
(tag `0xA019`, `uint32`) is sent in `CONFIG.request` from `config.c` (default
pool 64). The number of gNB antennas advertised in `CONFIG.request`
(`num_tx_ant`/`num_rx_ant`) sets the SRS estimate dimension Ng (e.g. 32 for a
single PE).

### 2.7 L2 code map

| file | role |
|---|---|
| `MACRLC_nr_paramdef.h` | `set_analog_beamforming = "dynamic_bfw"` → `SRS_DYNAMIC_BFW` |
| `nr_mac_gNB.h` | chest buffer pool types + `srs_chest_buf[]`, `num_srs_chest_buffers` |
| `gNB_scheduler_srs.c` | chest pool API; SRS-PDU `handle` stamping; REQUESTED marking |
| `gNB_scheduler_ulsch.c` | SRS.ind → `mark_ready`; T-Tracer chest log |
| `gNB_scheduler_dlsch.c` | `nr_sched_dynamic_bfw()` emitter + consumption gate |
| `gNB_scheduler.c` | per-slot call of the emitter |
| `gNB_scheduler_primitives.c` | free buffer on UE removal; SSB beam-index handling |
| `config.c` | `num_tx/rx_ant`, `NUM_SRS_CHEST_BUFFERS` TLV |
| `nfapi/…/nfapi_nr_interface_scf.h`, `nr_fapi_p7.c`, `fapi_nvIPC.c`, `fapi_vnf_p7.c` | BFW_CVI message + codec + transport |

---

## 3. L1 (NVIDIA Aerial / cuPHY) — overview

cuPHY is the FAPI PNF and does all the signal processing:

- **SRS channel estimation**: from the UL SRS U-plane it computes the channel
  estimate H[Ng, Nu, Np] (gNB antennas × UE ports × PRGs) and caches it in the
  chest buffer indicated by the SRS PDU `handle` (a pool sized by
  `NUM_SRS_CHEST_BUFFERS`). It also returns a normalized estimate in the
  `SRS.indication`.
- **Weight computation**: on a `DL_BFW_CVI.request`, cuPHY retrieves the cached
  estimate and computes the per-PRG DL beamforming weights (reciprocity), storing
  them in a slot-indexed BFW coefficient ring (hence L2's proactive 1-slot-ahead
  emission).
- **Application**: at PDSCH time the weights are emitted on the O-RAN fronthaul
  C-plane as Section Extension 11 (SE11) BFW, BFP-compressed.

cuPHY-side configuration (antenna count, SRS resources, chest mempool, and the
data-lake export of estimates/weights) lives in the cuphycontroller YAML.

---

## 4. O-RU — overview

The radio is an O-RAN 7.2x split unit. It receives the per-PRG beamforming
weights in the C-plane SE11 message (BFP-compressed I/Q + per-bundle exponent)
and applies them across its DL-TRX antenna elements to form the beam toward the
UE. No OAI/L2 code runs here; the RU only needs to support SE11 dynamic BFW and
the negotiated compression profile.

---

## 5. How to enable it (L2)

1. **Build** the gNB with Aerial support:
   ```
   cd cmake_targets && ./build_oai -w AERIAL --gNB --ninja
   ```
   (Dynamic BFW requires the SCF FAPI **222.10.04** wire format and `ENABLE_AERIAL`.)

2. **gNB configuration** (`.conf`):
   ```
   gNBs.[0].set_analog_beamforming = "dynamic_bfw";   # SRS_DYNAMIC_BFW mode
   gNBs.[0].do_SRS                 = "periodic";       # reciprocity needs SRS
   # configure the UE's SRS resources, pusch_AntennaPorts, and the carrier as usual
   ```
   The advertised gNB antenna count (`num_tx_ant`/`num_rx_ant` in `config.c`)
   must match the cuPHY/RU antenna setup (sets the SRS estimate Ng).

3. **Run** against an Aerial L1 (cuphycontroller) configured for SRS, the same
   antenna count, and a non-zero `num_srs_chest_buffers` pool.

4. **Verify** (L2 side):
   - The UE attaches and sounds SRS; `SRS.indication`s arrive and chest buffers
     reach READY (T-Tracer `GNB_MAC_SRS_CHANNEL_MATRIX` is populated).
   - With DL traffic to the UE, BFW_CVI requests are emitted each slot and cuPHY
     selects the dynamic SE11 weights (visible as varying beamIds in a fronthaul
     C-plane capture, vs. a fixed beam for static/broadcast).

---

## 6. Notes & limitations

- Single-cell, and one chest buffer per UE; the pool is sized at
  `NR_SRS_CHEST_BUF_POOL_SIZE` (64).
- The emitter currently requests weights for a single layer per UE; multi-layer
  / SU-MIMO is future work.
- A REQUESTED-state window around each sounding prevents `0x40
  SRS_CHEST_BUFF_BAD_STATE` rejections from cuPHY while a fresh estimate is in
  flight.
- The feature is inert unless `set_analog_beamforming = "dynamic_bfw"`; in any
  other mode the chest pool and emitter are not exercised.
