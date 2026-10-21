/* Per-member status codes (docs/PHASE0_SPEC.md section 6).
 *
 * Defined once here for the device code and mirrored by omnitool.status.Status;
 * tests/test_t8_status.py parses this file and asserts the two agree.
 */
#ifndef OMNITOOL_STATUS_H
#define OMNITOOL_STATUS_H

#define OMNITOOL_STATUS_NOT_RUN        (-1)
#define OMNITOOL_STATUS_SUCCESS        0
#define OMNITOOL_STATUS_MAX_STEPS      1
#define OMNITOOL_STATUS_STEP_TOO_SMALL 2
#define OMNITOOL_STATUS_NON_FINITE     3

#endif /* OMNITOOL_STATUS_H */
