// MIT License
//
// Copyright (c) 2022 Oleksandr Tkachenko
// Cryptography and Privacy Engineering Group (ENCRYPTO)
// TU Darmstadt, Germany
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "utility/constants.h"

namespace encrypto::motion::proto::garbled_circuit {

static constexpr std::size_t kGarbledControlBitsBitSize{5};

// the last bit is not used (extracted for garbling control bits)
static constexpr std::size_t kGarbledRowBitSize{kKappa / 2};
static constexpr std::size_t kGarbledRowByteSize{kGarbledRowBitSize / 8};

static constexpr std::size_t kGarbledTableBitSize{kGarbledRowBitSize * 3};
static constexpr std::size_t kGarbledTableByteSize{kGarbledTableBitSize / 8};

}  // namespace encrypto::motion::proto::garbled_circuit
