# SPDX-License-Identifier: LicenseRef-CSSL-1.0

# nr_srs_dl_ri.c is dependency-free math (no OAI headers), so the test links
# it directly instead of pulling in the full MAC.
add_executable(test_nr_srs_dl_ri
  test_nr_srs_dl_ri.c
  ../nr_srs_dl_ri.c)
target_link_libraries(test_nr_srs_dl_ri PRIVATE m)
add_dependencies(tests test_nr_srs_dl_ri)
add_test(NAME test_nr_srs_dl_ri
  COMMAND ./test_nr_srs_dl_ri)
