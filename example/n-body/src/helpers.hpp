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
        std::uniform_real_distribution<BaseType> dist(MASS_MIN, MASS_MAX);
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
        // most numbers should fall within the square that is plotted
        auto rd = std::random_device{};
        std::normal_distribution<BaseType> dist(
            (MAX_PARTICLE_POS + MIN_PARTICLE_POS) / 2.f,
            (MAX_PARTICLE_POS - MIN_PARTICLE_POS) / 4.f);

        for(auto i = 0u; i < xPositions.getExtents().x(); ++i)
        {
            xPositions[i] = dist(rd);
            yPositions[i] = dist(rd);
            zPositions[i] = dist(rd);
        }
    }

    /** @brief Helper function to generate a tangential velocity to prevent the whole set of particles moving in a
     * random direction.
     */
    Vec<BaseType, 3> randomTangentialVelocity(
        auto& rd,
        auto& dist,
        BaseType const x,
        BaseType const y,
        BaseType const z)
    {
        // Random vector
        BaseType const ax = dist(rd), ay = dist(rd), az = dist(rd);

        // Position vector
        BaseType const rx = x, ry = y, rz = z;
        BaseType const r_norm_sq = rx * rx + ry * ry + rz * rz;

        // Dot product of random vector and position vector
        BaseType const dot = ax * rx + ay * ry + az * rz;

        // Project random vector onto the tangential plane
        BaseType vx = ax - dot * rx / r_norm_sq;
        BaseType vy = ay - dot * ry / r_norm_sq;
        BaseType vz = az - dot * rz / r_norm_sq;

        return Vec{vz, vy, vx};
    }

    /** @brief Initialize the given x, y, and z-velocities with random values, given the positions. The positions are
     * used to generate tangential velocities to the coordinate center.
     * @note This is a host function.
     */
    template<concepts::MdSpan<BaseType> T_View>
    void initVelocities(
        T_View& xVelocities,
        T_View& yVelocities,
        T_View& zVelocities,
        T_View const& xPositions,
        T_View const& yPositions,
        T_View const& zPositions)
    {
        auto rd = std::random_device{};
        std::normal_distribution<BaseType> dist(VELOCITIES_MEAN, VELOCITIES_STDDEV);

        for(auto i = 0u; i < xVelocities.getExtents().x(); ++i)
        {
            auto const tangentialVelocity
                = randomTangentialVelocity(rd, dist, xPositions[i], yPositions[i], zPositions[i]);
            xVelocities[i] = tangentialVelocity.x();
            yVelocities[i] = tangentialVelocity.y();
            zVelocities[i] = tangentialVelocity.z();
        }
    }

#ifdef PNGWRITER_ENABLED
    struct Color
    {
        BaseType r;
        BaseType g;
        BaseType b;
    };

    template<concepts::MdSpan<Color> T_View>
    void initColors(T_View& colors)
    {
        auto rd = std::random_device{};
        std::uniform_real_distribution<BaseType> dist(COLOR_MIN, COLOR_MAX);

        for(auto i = 0u; i < colors.getExtents().x(); ++i)
        {
            colors[i].r = dist(rd);
            colors[i].g = dist(rd);
            colors[i].b = dist(rd);
        }
    }
#endif
} // namespace alpaka::example::nBody
