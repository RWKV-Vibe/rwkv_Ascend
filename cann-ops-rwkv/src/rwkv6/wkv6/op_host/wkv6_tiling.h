
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(wkv6TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);   
  TILING_DATA_FIELD_DEF(uint32_t, tileNumRemainLength);
  TILING_DATA_FIELD_DEF(uint32_t, totalHeads);   
  TILING_DATA_FIELD_DEF(uint32_t, T);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);  
  TILING_DATA_FIELD_DEF(uint32_t, HEAD_SIZE);   
  TILING_DATA_FIELD_DEF(uint32_t, HEAD_NUMS);   
  TILING_DATA_FIELD_DEF(bool, hasRemainer);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(wkv6, wkv6TilingData)
}
