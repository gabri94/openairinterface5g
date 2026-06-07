/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */
#include "nfapi/tests/nr_fapi_test.h"
#include "nr_fapi.h"
#include "nr_fapi_p7.h"

// Aerial vendor-extension DL/UL_BFW_CVI_REQUEST (msg IDs 0x90 / 0x91).
// DL and UL share the same on-the-wire layout, so a single round-trip test
// covers both — we run it twice with different message IDs.

static void fill_bfw_cvi_ue_config(nfapi_nr_bfw_cvi_ue_config_t *ue)
{
  ue->rnti = rand16_range(1, 0xFFEF);
  // bits 8..23 carry the SRS chest buffer index (0..1023)
  uint32_t buf_idx = rand16_range(0, 1023);
  ue->handle = (buf_idx & 0xFFFFu) << 8;
  ue->pdu_idx = rand16_range(0, 255);
  ue->gnb_ant_idx_start = rand8_range(0, 31);
  ue->gnb_ant_idx_end = rand8_range(ue->gnb_ant_idx_start, 63);
  ue->num_ue_ants = rand8_range(1, NFAPI_NR_MAX_BFW_CVI_UE_ANTS);
  for (uint8_t a = 0; a < ue->num_ue_ants; ++a) {
    ue->ue_ant_idx[a] = rand8_range(0, 3);
  }
}

static void fill_bfw_cvi_group_config(nfapi_nr_bfw_cvi_group_config_t *g)
{
  g->rb_start = rand16_range(0, 274);
  g->rb_size = rand16_range(1, 275);
  g->num_prgs = rand16_range(1, 273);
  g->prg_size = rand16_range(1, 16);
  g->num_ues = rand8_range(1, NFAPI_NR_MAX_BFW_CVI_UES_PER_GROUP);
  for (uint8_t u = 0; u < g->num_ues; ++u) {
    fill_bfw_cvi_ue_config(&g->ue_list[u]);
  }
}

static void fill_bfw_cvi_request(nfapi_nr_bfw_cvi_request_t *msg)
{
  msg->sfn = rand16_range(0, 1023);
  msg->slot = rand16_range(0, 159);
  msg->num_groups = rand8_range(1, NFAPI_NR_MAX_BFW_CVI_GROUPS);
  for (uint8_t g = 0; g < msg->num_groups; ++g) {
    fill_bfw_cvi_group_config(&msg->group_list[g]);
  }
}

static bool eq_bfw_cvi_ue_config(const nfapi_nr_bfw_cvi_ue_config_t *a,
                                 const nfapi_nr_bfw_cvi_ue_config_t *b)
{
  if (a->rnti != b->rnti || a->handle != b->handle || a->pdu_idx != b->pdu_idx
      || a->gnb_ant_idx_start != b->gnb_ant_idx_start || a->gnb_ant_idx_end != b->gnb_ant_idx_end
      || a->num_ue_ants != b->num_ue_ants) {
    return false;
  }
  for (uint8_t i = 0; i < a->num_ue_ants; ++i) {
    if (a->ue_ant_idx[i] != b->ue_ant_idx[i]) {
      return false;
    }
  }
  return true;
}

static bool eq_bfw_cvi_group_config(const nfapi_nr_bfw_cvi_group_config_t *a,
                                    const nfapi_nr_bfw_cvi_group_config_t *b)
{
  if (a->rb_start != b->rb_start || a->rb_size != b->rb_size || a->num_prgs != b->num_prgs
      || a->prg_size != b->prg_size || a->num_ues != b->num_ues) {
    return false;
  }
  for (uint8_t i = 0; i < a->num_ues; ++i) {
    if (!eq_bfw_cvi_ue_config(&a->ue_list[i], &b->ue_list[i])) {
      return false;
    }
  }
  return true;
}

static bool eq_bfw_cvi_request(const nfapi_nr_bfw_cvi_request_t *a,
                               const nfapi_nr_bfw_cvi_request_t *b)
{
  if (a->header.message_id != b->header.message_id || a->sfn != b->sfn || a->slot != b->slot
      || a->num_groups != b->num_groups) {
    return false;
  }
  for (uint8_t i = 0; i < a->num_groups; ++i) {
    if (!eq_bfw_cvi_group_config(&a->group_list[i], &b->group_list[i])) {
      return false;
    }
  }
  return true;
}

static void test_pack_unpack(nfapi_nr_bfw_cvi_request_t *req)
{
  uint8_t msg_buf[1024 * 1024];
  // Pack
  int pack_result = fapi_nr_p7_message_pack(req, msg_buf, sizeof(msg_buf), NULL);
  DevAssert(pack_result >= 0 + NFAPI_HEADER_LENGTH);
  req->header.message_length = pack_result;

  // Header unpack (SCTP peek simulation)
  fapi_message_header_t header = {0};
  int unpack_header_result =
      fapi_nr_p7_message_header_unpack(msg_buf, NFAPI_HEADER_LENGTH, &header, sizeof(header), 0);
  DevAssert(unpack_header_result >= 0);
  DevAssert(header.message_id == req->header.message_id);
  DevAssert(header.message_length == req->header.message_length);

  // Full unpack and compare
  nfapi_nr_bfw_cvi_request_t unpacked = {0};
  int unpack_result =
      fapi_nr_p7_message_unpack(msg_buf, header.message_length + NFAPI_HEADER_LENGTH, &unpacked, sizeof(unpacked), 0);
  DevAssert(unpack_result >= 0);
  DevAssert(eq_bfw_cvi_request(&unpacked, req));
}

static void run_for_message_id(nfapi_nr_phy_msg_type_e msg_id)
{
  nfapi_nr_bfw_cvi_request_t req = {.header.message_id = msg_id};
  fill_bfw_cvi_request(&req);
  test_pack_unpack(&req);

  // Edge case: maximum-fill request
  nfapi_nr_bfw_cvi_request_t big = {.header.message_id = msg_id};
  big.sfn = 1023;
  big.slot = 159;
  big.num_groups = NFAPI_NR_MAX_BFW_CVI_GROUPS;
  for (uint8_t g = 0; g < big.num_groups; ++g) {
    big.group_list[g].rb_start = 0;
    big.group_list[g].rb_size = 273;
    big.group_list[g].num_prgs = 273;
    big.group_list[g].prg_size = 1;
    big.group_list[g].num_ues = NFAPI_NR_MAX_BFW_CVI_UES_PER_GROUP;
    for (uint8_t u = 0; u < big.group_list[g].num_ues; ++u) {
      big.group_list[g].ue_list[u].rnti = (uint16_t)(0x4601u + u);
      big.group_list[g].ue_list[u].handle = ((uint32_t)(u + g * 16) & 0xFFFFu) << 8;
      big.group_list[g].ue_list[u].pdu_idx = u;
      big.group_list[g].ue_list[u].gnb_ant_idx_start = 0;
      big.group_list[g].ue_list[u].gnb_ant_idx_end = 31;
      big.group_list[g].ue_list[u].num_ue_ants = NFAPI_NR_MAX_BFW_CVI_UE_ANTS;
      for (uint8_t a = 0; a < big.group_list[g].ue_list[u].num_ue_ants; ++a) {
        big.group_list[g].ue_list[u].ue_ant_idx[a] = a;
      }
    }
  }
  test_pack_unpack(&big);
}

int main()
{
  fapi_test_init();

  run_for_message_id(NFAPI_NR_PHY_MSG_TYPE_DL_BFW_CVI_REQUEST);
  run_for_message_id(NFAPI_NR_PHY_MSG_TYPE_UL_BFW_CVI_REQUEST);

  return 0;
}
