# SRS-reciprocity dynamic downlink beamforming (Aerial)

This document describes the gNB **L2 (NR MAC)** support for SRS-reciprocity
dynamic downlink beamforming with the NVIDIA Aerial L1. The DL beamforming
weights do not come from a fixed beam table: L1 derives them per UE from the
uplink SRS channel estimate and applies them on the downlink (TDD reciprocity).

The feature is **Aerial-only** (`ENABLE_AERIAL`): OAI runs as the FAPI VNF
(L2) and NVIDIA Aerial / cuPHY as the PNF (L1). L2 never computes weights. It
tells L1 *which* stored SRS estimate to use and *when*, and L1 computes the
weights and applies them on the fronthaul.

> Scope: this document covers the **L2/MAC** design and how to enable it. The
> L1 (Aerial/cuPHY) and the O-RU are only summarized.

---

## 1. Flow

```
   UE                 L2 (OAI NR MAC)                L1 (Aerial/cuPHY)              O-RU
   │                       │                              │                           │
   │ ── SRS (periodic) ───────────────────────────────────────────────────────────► │
   │                       │ ① UL_TTI SRS PDU             │ ◄──── UL SRS U-plane ──── │
   │                       │   handle = uid << 8          │
   │                       │ ───────────────────────────► │ estimate H, store it in
   │                       │                              │ the UE's buffer (uid)
   │                       │ ② SRS.indication             │
   │      estimate READY   │ ◄─────────────────────────── │
   │                       │ ③ DL_BFW_CVI.request         │
   │                       │   every slot, all READY UEs  │ compute weights from H,
   │                       │ ───────────────────────────► │ store them in a slot ring
   │                       │ ④ DL_TTI PDSCH, no BeamId    │
   │                       │ ───────────────────────────► │ ── C-plane SE11 weights ─► │
   │ ◄───────────────────────────────────────────────────────────── beamformed PDSCH ── │
```

1. The UE sounds **periodic SRS**. The SRS PDU `handle` (bits 8-23) names the
   L1 buffer that will hold the UE's channel estimate: the UE's `uid`.
2. L1 estimates the channel and sends an `SRS.indication`. L2 marks the UE's
   estimate **READY**.
3. Every slot, L2 sends one **DL_BFW_CVI.request** (0x90) naming the buffer of
   each UE with a READY estimate. L1 computes the weights.
4. A PDSCH for a UE with a READY estimate carries no BeamId, so L1 applies the
   computed weights. They reach the O-RU as O-RAN C-plane Section Extension 11
   (SE11), and the RU beamforms.

---

## 2. L2 (NR MAC) design

Files are under `openair2/LAYER2/NR_MAC_gNB/` unless noted.

### 2.1 Enabling the mode

The mode is `bf_method = "dynamic"` (`BF_METHOD_DYNAMIC`, see
[beamforming.md](beamforming.md)) in the `MACRLCs` section. The SSB beams are
configured as for `"predefined"` and serve SSB, control and the PDSCH of a UE
without a READY estimate (RAR, Msg4). The weights come from SRS, so the UE
**must** sound periodic SRS (`do_SRS`).

### 2.2 SRS estimate buffers

L1 keeps one estimate buffer per UE, indexed by the UE's `uid`.
`nr_configure_srs()` (`gNB_scheduler_srs.c`) puts it in the SRS PDU
`handle` (`uid << 8`) for every Aerial SRS PDU, so the buffer never needs to be
allocated or released. A UE's `uid` is unique and stays the same for its
lifetime, and it is below `MAX_MOBILES_PER_GNB`, inside cuBB's default pool of
256 buffers per cell.

L2 only tracks the state of each UE's estimate, in `NR_UE_info_t.srs_chest`:

| state | set by | meaning |
|---|---|---|
| `NR_SRS_CHEST_NONE` | UE creation | no estimate yet |
| `NR_SRS_CHEST_PENDING` | `nr_configure_srs()` | SRS PDU in flight: L1 holds the buffer until its `SRS.indication` and rejects weight requests on it (`SRS_CHEST_BUFF_BAD_STATE`) |
| `NR_SRS_CHEST_READY` | `handle_nr_srs_measurements()` | estimate stored, with its antenna (`num_gnb_ant`) and port (`num_ue_ports`) counts |

`nr_srs_chest_ready(cell, UE)` is true when the cell uses `"dynamic"` and the
UE's estimate is READY. It gates both the weight request and the PDSCH BeamId.

### 2.3 The per-slot weight request

`nr_sched_dynamic_bfw()` (`gNB_scheduler_dlsch.c`) runs every slot from
`gNB_dlsch_ulsch_scheduler()`, after the DL scheduler.

cuBB stores the weights of a request in a slot-indexed ring, and a PDSCH in
slot N uses those requested for slot N-1, which must be exactly one slot old.
So the request is sent **every slot, for the current slot**: a PDSCH in any
slot finds weights from the previous slot, and L1 has a full slot to compute
them. The request does not consume the estimate, which stays READY until the
UE's next SRS PDU.

All READY UEs of the cell go in **one** request (one group, full BWP,
wideband), since cuBB rejects a second request for the same slot. The MAC fills
it in the slot's `NR_Sched_Rsp_t.DL_bfw_cvi_req`, and the VNF
(`nfapi/oai_integration/vnf/nfapi_nr_vnf.c`) sends it before the slot's
`DL_TTI.request`, like the other P7 messages.

### 2.4 PDSCH

`prepare_pdsch_pdu()` sets `dig_bf_interfaces = 0` (no BeamId, only `pm_idx`
per PRG) for a UE with a READY estimate, so L1 applies the computed weights.
Otherwise the PDSCH keeps the UE's SSB beam.

### 2.5 Several beams in one slot

With `"dynamic"`, `beam_allocation_procedure()` lets UEs on different SSB beams
share a slot instead of failing the allocation. Each UE's beam still reaches L1
per PDU. The UEs share one VRB map, so they are frequency-multiplexed, not
spatially multiplexed.

### 2.6 The BFW_CVI FAPI message (Aerial vendor extension)

- `NFAPI_NR_PHY_MSG_TYPE_DL_BFW_CVI_REQUEST = 0x90` (and UL `0x91`).
- `nfapi_nr_bfw_cvi_request_t` → `{sfn, slot, num_groups, group_list[]}`; each
  group → `{rb_start, rb_size, num_prgs, prg_size, ue_list[]}`; each UE →
  `{rnti, handle, pdu_idx, gnb_ant_idx_start/end, ue_ant_idx[]}`. On the wire
  each group starts with its size in bytes (`pdu_size`), as cuBB expects.
- Limits: `NFAPI_NR_MAX_BFW_CVI_GROUPS`, `…_UES_PER_GROUP`, `…_UE_ANTS`.

Pack/unpack is in `nfapi/open-nFAPI/fapi/src/nr_fapi_p7.c`, the nvIPC send
helper `oai_fapi_dl_bfw_cvi_req()` in `nfapi/oai_integration/aerial/fapi_nvIPC.c`.

`CONFIG.request` carries the Aerial vendor TLV `NUM_SRS_CHEST_BUFFERS` (0xA019,
`uint32`) only when `nr_mac_config_t.num_srs_chest_buffers` is set. A set
value must be at least `MAX_MOBILES_PER_GNB`, so that every `uid` has a buffer.

### 2.7 Code map

| file | role |
|---|---|
| `GNB_APP/gnb_config.c` | `bf_method = "dynamic"` → `BF_METHOD_DYNAMIC`, SSB beams |
| `nr_mac_gNB.h` | `nr_srs_chest_t`, `NR_UE_info_t.srs_chest` |
| `gNB_scheduler_srs.c` | SRS PDU `handle`, PENDING, `nr_srs_chest_ready()` |
| `gNB_scheduler_ulsch.c` | `SRS.indication` → READY |
| `gNB_scheduler_dlsch.c` | `nr_sched_dynamic_bfw()`, PDSCH without BeamId |
| `gNB_scheduler.c` | per-slot call of `nr_sched_dynamic_bfw()` |
| `NR_PHY_INTERFACE/NR_IF_Module.{c,h}` | `DL_bfw_cvi_req` in the slot's `NR_Sched_Rsp_t` |
| `nfapi/oai_integration/vnf/nfapi_nr_vnf.c` | sends the request before the `DL_TTI.request` |
| `gNB_scheduler_primitives.c` | several beams per slot |
| `config.c` | `NUM_SRS_CHEST_BUFFERS` TLV |
| `nfapi/…/nfapi_nr_interface_scf.h`, `nr_fapi_p7.c`, `fapi_nvIPC.c` | BFW_CVI message, codec, transport |

---

## 3. L1 (NVIDIA Aerial / cuPHY)

- **SRS channel estimation**: cuPHY estimates H (gNB antennas × UE ports ×
  PRGs) from the SRS U-plane, stores it in the buffer named by the SRS PDU
  `handle`, and returns a normalized copy in the `SRS.indication`.
- **Weight computation**: on a `DL_BFW_CVI.request`, cuPHY computes the per-PRG
  DL weights from the stored estimate into a slot-indexed ring.
- **Application**: at PDSCH time the weights go out on the fronthaul C-plane
  as SE11, BFP-compressed.

---

## 4. O-RU

An O-RAN 7.2x radio that supports SE11 dynamic weights. It applies the per-PRG
weights from the C-plane across its antenna elements.

---

## 5. How to enable it

1. **Build** the gNB with Aerial support:
   ```
   cd cmake_targets && ./build_oai -w AERIAL --gNB --ninja
   ```
   Dynamic beamforming needs cuBB with the SCF FAPI 222.10.04 wire format.

2. **gNB configuration**:
   ```
   gNBs = ({ ... do_SRS = "periodic"; ... });
   MACRLCs = ({ ... bf_method = "dynamic"; ssb_beams = (0); ... });
   Aerial = { num_tx_ant = 32; num_rx_ant = 32; };
   ```
   `Aerial.num_tx_ant`/`num_rx_ant` are the TXRUs advertised to L1, which also
   set the number of gNB antennas in the SRS estimate. They must match the
   cuPHY and RU setup.

3. **Run** against an Aerial L1 configured for SRS with the same antenna count.

4. **Verify**:
   - The UE attaches and sounds SRS, and `SRS.indication`s arrive.
   - With DL traffic, one `DL_BFW_CVI.request` goes out per slot and a
     fronthaul C-plane capture shows SE11 weights for the UE's PDSCH instead of
     its static beam.

---

## 6. Limitations

- One weight group per request (wideband, full BWP), so at most
  `NFAPI_NR_MAX_BFW_CVI_UES_PER_GROUP` UEs per slot get weights.
- UEs on different beams share a slot by frequency multiplexing only (no
  MU-MIMO).
- Only DL weights are requested (`0x90`); the UL request (`0x91`) is defined
  but not sent.
- With any other `bf_method`, the estimate state and the weight request are
  not used.
