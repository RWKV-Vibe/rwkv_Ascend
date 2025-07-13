
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_WKV6_H_
#define ACLNN_WKV6_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnwkv6GetWorkspaceSize
 * parameters :
 * k : required
 * v : required
 * w : required
 * r : required
 * u : required
 * hi : required
 * oOut : required
 * hoOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnwkv6GetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *w,
    const aclTensor *r,
    const aclTensor *u,
    const aclTensor *hi,
    const aclTensor *oOut,
    const aclTensor *hoOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnwkv6
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnwkv6(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
