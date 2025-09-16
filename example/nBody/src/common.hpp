/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/CVec.hpp>

#include <cstdint>

namespace alpaka::example::nBody
{
    using BaseType = float;
    using IdxType = std::uint32_t;

    constexpr IdxType operator""_idx(unsigned long long const n)
    {
        return static_cast<IdxType>(n);
    }

    constexpr BaseType operator""_bt(long double const v)
    {
        return static_cast<BaseType>(v);
    }

    constexpr auto flopsRequiredPerTimeStep(IdxType const numElements)
    {
        return
            // 20 flops per particle-particle acceleration calculation
            20 * numElements * numElements +
            // 6 flops for position and velocity update per element
            6 * numElements;
    }

    // default values for a run
    constexpr IdxType defaultTimeSteps = 1000;
    constexpr IdxType defaultNumParticles = 512;
    constexpr BaseType defaultDt = 0.001_bt;

    // gravity constant
    constexpr BaseType GRAV = 6.674e-11_bt;

    // softening factor that is added to particle distance to prevent too large forces
    constexpr BaseType EPS = 4._bt;

    // every this many simulation steps, a png will be written to disk
    constexpr int pngStepSize = 10;

    // the screen width and height if pngs are written
    constexpr IdxType screenWidth = 1000;
    constexpr IdxType screenHeight = 1000;

    // when a particle is closer to the camera than this, it will not be shown when pngs are written
    constexpr BaseType zClipNear = 100._bt;

    // minimum and maximum color values for the particle colors when writing pngs
    constexpr BaseType colorMin = 0.2_bt;
    constexpr BaseType colorMax = 1.0_bt;

    // minimum and maximum particle positions, same for x, y, and z coordinates
    // these are only approximate when a normal distribution is used, outliers can exist
    constexpr BaseType minParticlePos = -1500._bt;
    constexpr BaseType maxParticlePos = 1500._bt;

    // minimum and maximum mass for particles
    constexpr BaseType massMin = 1e6_bt;
    constexpr BaseType massMax = 1e7_bt;

    // mean and stddev of velocities, same in every coordinate
    constexpr BaseType velocitiesMean = 0._bt;
    constexpr BaseType velocitiesStdDev = 700._bt;

} // namespace alpaka::example::nBody
