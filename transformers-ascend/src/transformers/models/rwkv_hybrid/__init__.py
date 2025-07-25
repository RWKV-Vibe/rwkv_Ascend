# Copyright 2024 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_rwkv_hybrid": ["RwkvHybridConfig"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_rwkv_hybrid"] = [
        "RwkvHybridForCausalLM",
        "RwkvHybridModel",
        "RwkvHybridPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_rwkv_hybrid import RwkvHybridConfig
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_rwkv_hybrid import (
            RwkvHybridForCausalLM,
            RwkvHybridModel,
            RwkvHybridPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
