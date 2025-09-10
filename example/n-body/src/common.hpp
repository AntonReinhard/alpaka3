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

    constexpr IdxType SCREEN_WIDTH = 1000;
    constexpr IdxType SCREEN_HEIGHT = 1000;

    constexpr BaseType MIN_PARTICLE_POS = -1500.f;
    constexpr BaseType MAX_PARTICLE_POS = 1500.f;

    constexpr BaseType Z_CLIP_NEAR = 100.f;

} // namespace alpaka::example::nBody
