#ifndef WKV7GRAD_PROTO_H_
#define WKV7GRAD_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(wkv7grad)
    .INPUT(k, ge::TensorType::ALL())
    .INPUT(v, ge::TensorType::ALL())
    .INPUT(w, ge::TensorType::ALL())
    .INPUT(r, ge::TensorType::ALL())
    .INPUT(a, ge::TensorType::ALL())
    .INPUT(b, ge::TensorType::ALL())
    .INPUT(h, ge::TensorType::ALL())
    .INPUT(o, ge::TensorType::ALL())
    .OUTPUT(dk, ge::TensorType::ALL())
    .OUTPUT(dv, ge::TensorType::ALL())
    .OUTPUT(dw, ge::TensorType::ALL())
    .OUTPUT(dr, ge::TensorType::ALL())
    .OUTPUT(da, ge::TensorType::ALL())
    .OUTPUT(db, ge::TensorType::ALL())
    .OUTPUT(dh, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(wkv7grad);

}

#endif
