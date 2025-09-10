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
    constexpr int pngStepSize = 10;

    // the screen width and height if pngs are written
    constexpr IdxType screenWidth = 1000;
    constexpr IdxType screenHeight = 1000;

    // when a particle is closer to the camera than this, it will not be shown when pngs are written
    constexpr BaseType zClipNear = 100.f;

    // minimum and maximum color values for the particle colors when writing pngs
    constexpr BaseType colorMin = 0.2f;
    constexpr BaseType colorMax = 1.0f;

    // minimum and maximum particle positions, same for x, y, and z coordinates
    // these are only approximate when a normal distribution is used, outliers can exist
    constexpr BaseType minParticlePos = -1500.f;
    constexpr BaseType maxParticlePos = 1500.f;

    // minimum and maximum mass for particles
    constexpr BaseType massMin = 1e6f;
    constexpr BaseType massMax = 1e7f;

    // mean and stddev of velocities, same in every coordinate
    constexpr BaseType velocitiesMean = 0.f;
    constexpr BaseType velocitiesStdDev = 700.f;

    constexpr IdxType operator""_idx(unsigned long long const n)
    {
        return static_cast<IdxType>(n);
    }
} // namespace alpaka::example::nBody
