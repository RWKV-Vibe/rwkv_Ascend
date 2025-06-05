#include "wkv4_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        TilingData tiling;

        int32_t ubLimit = 176 * 1024;
        const gert::Shape xShape = context->GetInputShape(2)->GetStorageShape();
        int32_t batchSize = (int32_t)xShape.GetDim(0);
        int32_t ctxLen = (int32_t)xShape.GetDim(1);
        int32_t nEmbd = (int32_t)xShape.GetDim(2);
        tiling.set_batchSize(batchSize);
        tiling.set_ctxLen(ctxLen);
        tiling.set_nEmbd(nEmbd);
        tiling.set_ubLimit(ubLimit);

        size_t usrSize = 0;
        size_t sysWorkspaceSize = 16 * 1024 * 1024;
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = sysWorkspaceSize + usrSize;

        context->SetBlockDim(batchSize);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    ge::graphStatus InferShape2(gert::InferShapeContext* context) {
        auto k_shape = context->GetInputShape(2);
        auto m_shape = context->GetInputShape(4);
        auto out_shape = context->GetOutputShape(0);
        auto m_out_shape = context->GetOutputShape(1);
        auto p_out_shape = context->GetOutputShape(2);
        auto q_out_shape = context->GetOutputShape(3);
        *out_shape = *k_shape;
        *m_out_shape = *m_shape;
        *p_out_shape = *m_shape;
        *q_out_shape = *m_shape;
        return GRAPH_SUCCESS;
    }

    ge::graphStatus InferDataType2(gert::InferDataTypeContext* context) {
        const ge::DataType w_datatype = context->GetInputDataType(0);
        context->SetOutputDataType(0, w_datatype);
        context->SetOutputDataType(1, w_datatype);
        context->SetOutputDataType(2, w_datatype);
        context->SetOutputDataType(3, w_datatype);
        return GRAPH_SUCCESS;
    }
}


namespace ops {
    class Wkv : public OpDef {
    public:
        explicit Wkv(const char* name) : OpDef(name) {
            this->Input("w")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND });
            this->Input("u")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND });
            this->Input("k")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND });
            this->Input("v")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND });
            this->Input("m")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND });
            this->Input("p")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND });
            this->Input("q")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND });
            this->Output("o")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND });
            this->Output("mo")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND });
            this->Output("po")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND });
            this->Output("qo")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND });

            this->SetInferShape(ge::InferShape2)
                 .SetInferDataType(ge::InferDataType2);

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
    OP_ADD(wkv4);
}