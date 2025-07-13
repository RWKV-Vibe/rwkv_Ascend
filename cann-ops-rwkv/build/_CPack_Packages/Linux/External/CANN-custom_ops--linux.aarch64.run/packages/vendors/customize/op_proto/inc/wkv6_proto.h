#ifndef WKV6_PROTO_H_
#define WKV6_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(wkv6)
    .INPUT(k, ge::TensorType::ALL())
    .INPUT(v, ge::TensorType::ALL())
    .INPUT(w, ge::TensorType::ALL())
    .INPUT(r, ge::TensorType::ALL())
    .INPUT(u, ge::TensorType::ALL())
    .INPUT(hi, ge::TensorType::ALL())
    .OUTPUT(o, ge::TensorType::ALL())
    .OUTPUT(ho, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(wkv6);

}

#endif
