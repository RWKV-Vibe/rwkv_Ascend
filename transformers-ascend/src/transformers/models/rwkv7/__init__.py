# -*- coding: utf-8 -*-
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
    is_torch_available,
)

_import_structure = {
    "configuration_rwkv7": ["RWKV7Config"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_rwkv7"] = [
        "RWKV7ForCausalLM",
        "RWKV7Model",
    ]
    _import_structure["tokenization_rwkv7"] = [
        "RWKV7Tokenizer",
    ]

if TYPE_CHECKING:
    from .configuration_rwkv7 import RWKV7Config

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_rwkv7 import (
            RWKV7ForCausalLM,
            RWKV7Model,
        )
        from .tokenization_rwkv7 import RWKV7Tokenizer
        
        # Register the model classes with Auto classes
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
        AutoConfig.register(RWKV7Config.model_type, RWKV7Config, True)
        AutoModel.register(RWKV7Config, RWKV7Model, True)
        AutoModelForCausalLM.register(RWKV7Config, RWKV7ForCausalLM, True)

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)

    # Register the model classes with Auto classes when not in TYPE_CHECKING mode
#     from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
#     AutoConfig.register("rwkv7", RWKV7Config, True)
#     AutoModel.register(RWKV7Config, RWKV7Model, True)
#     AutoModelForCausalLM.register(RWKV7Config, RWKV7ForCausalLM, True)

# __all__ = ['RWKV7Config', 'RWKV7ForCausalLM', 'RWKV7Model']