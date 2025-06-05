/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lock.cpp
 * \brief
 */
#include "lock.h"

namespace optiling {
void RWLock::rdlock()
{
    std::unique_lock<std::mutex> lck(_mtx);
    _waiting_readers += 1;
    _read_cv.wait(lck, [&]() { return _waiting_writers == 0 && _status >= 0; });
    _waiting_readers -= 1;
    _status += 1;
}

void RWLock::wrlock()
{
    std::unique_lock<std::mutex> lck(_mtx);
    _waiting_writers += 1;
    _write_cv.wait(lck, [&]() { return _status == 0; });
    _waiting_writers -= 1;
    _status = -1;
}

void RWLock::unlock()
{
    std::unique_lock<std::mutex> lck(_mtx);
    if (_status == -1) {
        _status = 0;
    } else {
        _status -= 1;
    }
    if (_waiting_writers > 0) {
        if (_status == 0) {
            _write_cv.notify_one();
        }
    } else {
        _read_cv.notify_all();
    }
}
} // namespace optiling
