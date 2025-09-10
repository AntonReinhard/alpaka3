/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <cstdint>

namespace alpaka::example::nBody
{
    using BaseType = float;
    using IdxType = std::uint32_t;

    // gravity constant
    constexpr BaseType GRAV = 6.674e-11;

    // softening factor that is added to particle distance to prevent too large forces
    constexpr BaseType EPS = 4.f;

} // namespace alpaka::example::nBody
