
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_WKV7GRAD_H_
#define ACLNN_WKV7GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnwkv7gradGetWorkspaceSize
 * parameters :
 * k : required
 * v : required
 * w : required
 * r : required
 * a : required
 * b : required
 * h : required
 * o : required
 * dkOut : required
 * dvOut : required
 * dwOut : required
 * drOut : required
 * daOut : required
 * dbOut : required
 * dhOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnwkv7gradGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *w,
    const aclTensor *r,
    const aclTensor *a,
    const aclTensor *b,
    const aclTensor *h,
    const aclTensor *o,
    const aclTensor *dkOut,
    const aclTensor *dvOut,
    const aclTensor *dwOut,
    const aclTensor *drOut,
    const aclTensor *daOut,
    const aclTensor *dbOut,
    const aclTensor *dhOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnwkv7grad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnwkv7grad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
