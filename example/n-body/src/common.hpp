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

    // every this many simulation steps, a png will be written to disk
    constexpr int PNG_STEP_SIZE = 10;

    // the screen width and height if pngs are written
    constexpr IdxType SCREEN_WIDTH = 1000;
    constexpr IdxType SCREEN_HEIGHT = 1000;

    // when a particle is closer to the camera than this, it will not be shown when pngs are written
    constexpr BaseType Z_CLIP_NEAR = 100.f;

    // minimum and maximum color values for the particle colors when writing pngs
    constexpr BaseType COLOR_MIN = 0.2f;
    constexpr BaseType COLOR_MAX = 1.0f;

    // minimum and maximum particle positions, same for x, y, and z coordinates
    // these are only approximate when a normal distribution is used, outliers can exist
    constexpr BaseType MIN_PARTICLE_POS = -1500.f;
    constexpr BaseType MAX_PARTICLE_POS = 1500.f;

    // minimum and maximum mass for particles
    constexpr BaseType MASS_MIN = 1e6f;
    constexpr BaseType MASS_MAX = 1e7f;

    // mean and stddev of velocities, same in every coordinate
    constexpr BaseType VELOCITIES_MEAN = 0.f;
    constexpr BaseType VELOCITIES_STDDEV = 700.f;
} // namespace alpaka::example::nBody
