#include "register/tilingdata_base.h"

namespace optiling
{
    BEGIN_TILING_DATA_DEF(TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, ub_size);
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, T);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(WkvGrad, TilingData)
}