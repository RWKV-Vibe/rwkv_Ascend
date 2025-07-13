#include <string.h>
#include "graph/types.h"
#include "aclnn_wkv6.h"

namespace {
typedef struct {
    uint32_t id;
    const char *funcName;
    bool hasReg;
} NnopbaseDfxId;
typedef struct {
    ge::DataType dtype;
    ge::Format format;
} TensorDesc;
typedef struct {
    TensorDesc *inputsDesc;
    size_t inputsNum;
    TensorDesc *outputsDesc;
    size_t outputsNum;
} SupportInfo;
typedef struct {
    SupportInfo *supportInfo;
    size_t num;
} OpSocSupportInfo;
typedef struct {
    OpSocSupportInfo *socSupportInfo;
    size_t num;
} OpSupportList;
enum SocType {
    SOC_VERSION_ASCEND910A = 1,
    SOC_VERSION_ASCEND910B,
    SOC_VERSION_ASCEND910_93,
    SOC_VERSION_ASCEND910_95,
    SOC_VERSION_ASCEND310P,
    SOC_VERSION_ASCEND310B,
    SOC_VERSION_BS9SX1A,
    SOC_VERSION_ASCEND610Lite,
    SOC_VERSION_ASCEND910_55,
    SOC_VERSION_MC61AM21A
};
enum NnopbaseAttrDtype {
    kNnopbaseBool = 0U,
    kNnopbaseFloat,
    kNnopbaseInt,
    kNnopbaseString,
    kNnopbaseAttrEnd
};
uint32_t socSupportList[] = {SOC_VERSION_ASCEND910B};
uint32_t socSupportListLen = 1;

TensorDesc inputDesc0_0[6] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND}};
TensorDesc outputDesc0_0[2] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND}};
SupportInfo list0_0 = {inputDesc0_0, 6, outputDesc0_0, 2};
SupportInfo supportInfo0[1] = {list0_0};
OpSocSupportInfo socSupportInfo0= {supportInfo0, 1};

OpSocSupportInfo opSocSupportList[1] = {socSupportInfo0};
OpSupportList supportList = {opSocSupportList, 1};

[[maybe_unused]] uint32_t NNOPBASE_wkv6 = 0U;
} // namespace

extern void NnopbaseOpLogE(const aclnnStatus code, const char *const expr);

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus NnopbaseCreateExecutorSpace(void **space);
extern void *NnopbaseGetExecutor(void *space, const char *opType, char *inputsDesc, uint32_t inputNum,
                                 char *outputsDesc, uint32_t outputNum, char *attrsDesc, uint32_t attrsNum);
extern aclnnStatus NnopbaseAddInput(void *executor, const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddIgnoreContinuesInput(void *executor,
                                                   const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddIntArrayInput(void *executor, const aclIntArray *array, const uint32_t index);
extern aclnnStatus NnopbaseAddBoolArrayInput(void *executor, const aclBoolArray *array, const uint32_t index);
extern aclnnStatus NnopbaseAddFloatArrayInput(void *executor, const aclFloatArray *array, const uint32_t index);
extern aclnnStatus NnopbaseAddOutput(void *executor, const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddDynamicInput(void *executor, const aclTensorList *tensor_list, const uint32_t index);
extern aclnnStatus NnopbaseAddDynamicOutput(void *executor, const aclTensorList *tensor_list, const uint32_t index);
extern aclnnStatus NnopbaseAddAttrWithDtype(void *executor, void *attrAddr, size_t attrLen, const size_t index, const NnopbaseAttrDtype dtype);
extern aclnnStatus NnopbaseAddIntArrayAttr(void *executor, const aclIntArray* array, const size_t index);
extern aclnnStatus NnopbaseAddFloatArrayAttr(void *executor, const aclFloatArray* array, const size_t index);
extern aclnnStatus NnopbaseAddBoolArrayAttr(void *executor, const aclBoolArray* array, const size_t index);
extern aclnnStatus NnopbaseAddArrayAttrWithDtype(void *executor, void *array, const size_t len, const size_t elementSize, const size_t index, const NnopbaseAttrDtype dtype);
extern uint64_t NnopbaseMsprofSysTime();
extern aclnnStatus NnopbaseAddTilingId(void *executor, NnopbaseDfxId *tilingId);
extern void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);
extern aclnnStatus NnopbaseRunForWorkspace(void *executor, uint64_t *workspaceLen);
extern aclnnStatus NnopbaseRunWithWorkspace(void *executor, aclrtStream stream, void *workspace, uint64_t workspaceSize);
extern aclnnStatus NnopbaseAddSupportList(void *executor, OpSupportList *list, uint32_t *socSupportList, size_t socSupportListLen);
extern aclnnStatus NnopbaseAddScalarInput(void *executor, const aclScalar *scalar, const uint32_t index, const int32_t srcIndex, const ge::DataType dtype);
extern aclnnStatus NnopbaseAddScalarListInput(void *executor, const aclScalarList *scalarList, const uint32_t index, const int32_t srcIndex, const ge::DataType dtype);
extern void NnopbaseAddOpTypeId(void *executor, const uint32_t opTypeId);
extern aclnnStatus __attribute__((weak)) NnopbaseAddParamName(void *executor, const uint32_t index, const char *name, const bool isInput);
extern aclnnStatus __attribute__((weak)) NnopbaseSetFormatMatchMode(void *executor, const uint32_t mode);
extern aclnnStatus NnopbaseSetRef(void *executor, const size_t inputIrIdx, const size_t outputIrIdx);

#define ACLNN_SUCCESS  0
#define ACLNN_ERR_PARAM_NULLPTR 161001
#define ACLNN_ERR_PARAM_INVALID 161002

#define NNOPBASE_ASSERT_OK_RETVAL(v)                                    \
    do {                                                                \
        const aclnnStatus _chk_stutus = (v);                            \
        if (_chk_stutus != ACLNN_SUCCESS) {                             \
            NnopbaseOpLogE(_chk_stutus, #v);                            \
            return _chk_stutus;                                         \
        }                                                               \
    } while (false)

#define NNOPBASE_ASSERT_NOTNULL_RETVAL(v)                               \
    do {                                                                \
        if ((v) == nullptr) {                                           \
            NnopbaseOpLogE(ACLNN_ERR_PARAM_NULLPTR, #v " != nullptr");  \
            return ACLNN_ERR_PARAM_NULLPTR;                             \
        }                                                               \
    } while (false)

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
    aclOpExecutor **executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    static NnopbaseDfxId tilingId = {0x60000, "aclnnwkv6Tiling", false};
    void *nnopExecutor;
    static void *executorSpace = NULL;
    const char *opType = "wkv6";
    char inputDesc[] = {1, 1, 1, 1, 1, 1};
    char outputDesc[] = {1, 1};
    char attrDesc[] = {};

    NNOPBASE_ASSERT_NOTNULL_RETVAL(k);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(v);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(w);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(r);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(u);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hi);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(oOut);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hoOut);

    if (!executorSpace) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCreateExecutorSpace(&executorSpace));
    }
    nnopExecutor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                       sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    NNOPBASE_ASSERT_NOTNULL_RETVAL(nnopExecutor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    *executor = reinterpret_cast<aclOpExecutor *>(nnopExecutor);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddTilingId(*executor, &tilingId));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, k, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, v, 1));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, w, 2));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, r, 3));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, u, 4));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, hi, 5));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, oOut, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, hoOut, 1));
    if (NnopbaseAddParamName != NULL) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 0, "k", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 1, "v", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 2, "w", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 3, "r", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 4, "u", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 5, "hi", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 0, "oOut", false));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 1, "hoOut", false));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddSupportList(*executor, &supportList, socSupportList, socSupportListLen));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseRunForWorkspace(*executor, workspaceSize));
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnwkv6(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceSize));
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
