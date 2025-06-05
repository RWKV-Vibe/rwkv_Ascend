#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(TilingData)
        TILING_DATA_FIELD_DEF(int32_t, batchSize);
        TILING_DATA_FIELD_DEF(int32_t, ctxLen);
        TILING_DATA_FIELD_DEF(int32_t, ubLimit);
        TILING_DATA_FIELD_DEF(int32_t, nEmbd);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(Wkv, TilingData)
}
