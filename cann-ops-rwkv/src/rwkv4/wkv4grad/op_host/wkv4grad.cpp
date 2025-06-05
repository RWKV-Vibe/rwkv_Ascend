#include "wkv4grad_tiling.h"
#include "register/op_def_registry.h"

namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {

        TilingData tiling;
        const gert::StorageShape *k_shape = context->GetInputShape(2);

        int32_t B, T, C;
        B = k_shape->GetStorageShape().GetDim(0);
        T = k_shape->GetStorageShape().GetDim(1);
        C = k_shape->GetStorageShape().GetDim(2);

        tiling.set_B(B);
        tiling.set_T(T);
        tiling.set_C(C);

        uint64_t ubSizePlatForm = 190 * 1024;
        uint32_t coreNum = 32;

        ubSizePlatForm -= 5 * 1024;

        tiling.set_ub_size(ubSizePlatForm);
        context->SetBlockDim(coreNum);

        size_t usrSize = 3 * B * T * C * sizeof(float);
        size_t sysWorkspaceSize = 16 * 1024 * 1024;
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = sysWorkspaceSize + usrSize;

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge
{
    ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        auto w_shape = context->GetInputShape(0);
        auto k_shape = context->GetInputShape(2);
        auto gw_shape = context->GetOutputShape(0);
        auto gu_shape = context->GetOutputShape(1);
        auto gk_shape = context->GetOutputShape(2);
        auto gv_shape = context->GetOutputShape(3);
        *gw_shape = *w_shape;
        *gu_shape = *w_shape;
        *gk_shape = *k_shape;
        *gv_shape = *k_shape;
        return GRAPH_SUCCESS;
    }


    ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        const ge::DataType w_datatype = context->GetInputDataType(0);
        context->SetOutputDataType(0, w_datatype);
        context->SetOutputDataType(1, w_datatype);
        context->SetOutputDataType(2, w_datatype);
        context->SetOutputDataType(3, w_datatype);
        return GRAPH_SUCCESS;
    }
}

namespace ops
{
    class WkvGrad : public OpDef
    {
    public:
        WkvGrad(const char *name) : OpDef(name)
        {
            this->Input("w")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("u")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("k")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("v")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("gy")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("gw")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("gu")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("gk")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("gv")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->SetInferShape(ge::InferShape)
                .SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc);

            OpAICoreConfig aicConfig;
            aicConfig.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .PrecisionReduceFlag(false);

            this->AICore().AddConfig("ascend910");
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(wkv4grad);
}

