/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "common.hpp"

#include <alpaka/alpaka.hpp>

#include <random>

namespace alpaka::example::nBody
{
    /** @brief Initialize the given 1-dimensional MdSpan with random masses.
     * @note This is a host function.
     */
    template<concepts::MdSpan<BaseType> T_View>
    void initMasses(T_View& masses)
    {
        auto rd = std::random_device{};
        std::uniform_real_distribution<BaseType> dist(1e5f, 1e9f);
        for(auto i = 0u; i < masses.getExtents().x(); ++i)
        {
            masses[i] = dist(rd);
        }
    }

    /** @brief Initialize the given x, y, and z-positions with random values.
     * @note This is a host function.
     */
    template<concepts::MdSpan<BaseType> T_View>
    void initPositions(T_View& xPositions, T_View& yPositions, T_View& zPositions)
    {
        auto rd = std::random_device{};
        std::uniform_real_distribution<BaseType> dist(-1e3f, 1e3f);

        for(auto i = 0u; i < xPositions.getExtents().x(); ++i)
        {
            xPositions[i] = dist(rd);
            yPositions[i] = dist(rd);
            zPositions[i] = dist(rd);
        }
    }

    /** @brief Initialize the given x, y, and z-velocities with random values.
     * @note This is a host function.
     */
    template<concepts::MdSpan<BaseType> T_View>
    void initVelocities(T_View& xVelocities, T_View& yVelocities, T_View& zVelocities)
    {
        auto rd = std::random_device{};
        std::uniform_real_distribution<BaseType> dist(-1.f, 1.f);

        for(auto i = 0u; i < xVelocities.getExtents().x(); ++i)
        {
            xVelocities[i] = dist(rd);
            yVelocities[i] = dist(rd);
            zVelocities[i] = dist(rd);
        }
    }
} // namespace alpaka::example::nBody
