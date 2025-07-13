/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <set>
#include "ms_extension.h"

using BaseTensor = mindspore::tensor::BaseTensor;
using BaseTensorPtr = mindspore::tensor::BaseTensorPtr;
using TypeId = mindspore::TypeId;
using PyBoostUtils = mindspore::kernel::pyboost::PyBoostUtils;


namespace mindspore {
std::tuple<BaseTensorPtr, BaseTensorPtr> wkv7(const BaseTensorPtr &k, const BaseTensorPtr &v, const BaseTensorPtr &w, const BaseTensorPtr &r, const BaseTensorPtr &a, const BaseTensorPtr &b, const BaseTensorPtr &hi) {
  mindspore::runtime::ProfilerRecorder profiler(mindspore::runtime::ProfilerModule::kPynative,
                                                mindspore::runtime::ProfilerEvent::kRunOp, "wkv7");
  ShapeVector o_shape = k->shape();
  ShapeVector ho_shape = hi->shape();
  auto stream_id = PyBoostUtils::cur_stream_id();
  auto device_context = mindspore::runtime::OpRunner::GetDeviceContext("Ascend");
  // auto type_id = dtype.has_value() ? static_cast<TypeId>(dtype.value()) : x->data_type();
  // auto axis_vec = axis.has_value() ? axis.value() : std::vector<int64_t>();
  auto out = std::make_shared<BaseTensor>(k->data_type(), o_shape);
  auto ho = std::make_shared<BaseTensor>(k->data_type(), ho_shape);

  // auto y = kernel::pyboost::abs(x);

  // create DeviceAddress for inputs and outputs
  PyBoostUtils::PrepareOpInputs(device_context, stream_id, k);
  PyBoostUtils::PrepareOpInputs(device_context, stream_id, v);
  PyBoostUtils::PrepareOpInputs(device_context, stream_id, w);
  PyBoostUtils::PrepareOpInputs(device_context, stream_id, r);
  PyBoostUtils::PrepareOpInputs(device_context, stream_id, a);
  PyBoostUtils::PrepareOpInputs(device_context, stream_id, b);
  PyBoostUtils::PrepareOpInputs(device_context, stream_id, hi);
  PyBoostUtils::PrepareOpOutputs(device_context, stream_id, {out});
  PyBoostUtils::PrepareOpOutputs(device_context, stream_id, {ho});

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([=]() {
    MS_LOG(DEBUG) << "Run device task aclnnwkv7 start";
    // malloc device memory for inputs and outputs
    PyBoostUtils::MallocOpInputs(device_context, k);
    PyBoostUtils::MallocOpInputs(device_context, v);
    PyBoostUtils::MallocOpInputs(device_context, w);
    PyBoostUtils::MallocOpInputs(device_context, r);
    PyBoostUtils::MallocOpInputs(device_context, a);
    PyBoostUtils::MallocOpInputs(device_context, b);
    PyBoostUtils::MallocOpInputs(device_context, hi);
    PyBoostUtils::MallocOpOutputs(device_context, {out});
    PyBoostUtils::MallocOpOutputs(device_context, {ho});
    // launch aclnn op
    LAUNCH_ACLNN(aclnnwkv7, device_context, stream_id, k, v, w, r, a, b, hi, out, ho);
    MS_LOG(DEBUG) << "Run device task aclnnwkv7 end";
  }));
  return std::make_tuple(out, ho);
}
}  // namespace mindspore

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("wkv7", &mindspore::wkv7, "aclnnwkv7", pybind11::arg("k"),pybind11::arg("v"),pybind11::arg("w"),pybind11::arg("r"),pybind11::arg("a"),pybind11::arg("b"),pybind11::arg("hi"));
}
