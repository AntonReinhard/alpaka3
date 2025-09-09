/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <cmath>
#include <random>

namespace alpaka::example::nBody
{
    using BaseType = float;
    using IdxType = uint32_t;

    // gravity constant
    constexpr BaseType GRAV = 6.674e-11;

    // softening factor that is added to r to prevent too large forces
    constexpr BaseType EPS = 1e-6;

    struct Particle
    {
        BaseType mass;
        BaseType xPos;
        BaseType yPos;
        BaseType zPos;
        BaseType xVel;
        BaseType yVel;
        BaseType zVel;

        constexpr void update(Particle const& other)
        {
            BaseType const r_x = other.xPos - xPos;
            BaseType const r_y = other.yPos - yPos;
            BaseType const r_z = other.zPos - zPos;

            BaseType const distSqr = r_x * r_x + r_y * r_y + r_z * r_z + EPS;
            BaseType const distSixth = distSqr * distSqr * distSqr;
            BaseType const invDistCube = 1.0f / math::sqrt(distSixth);
            BaseType const a = other.mass * invDistCube; // acceleration

            xVel += a * r_x;
            yVel += a * r_y;
            zVel += a * r_z;
        }
    };

    struct RefParticle
    {
        BaseType& mass;
        BaseType& xPos;
        BaseType& yPos;
        BaseType& zPos;
        BaseType& xVel;
        BaseType& yVel;
        BaseType& zVel;

        RefParticle& operator=(Particle const& p)
        {
            mass = p.mass;
            xPos = p.xPos;
            yPos = p.yPos;
            zPos = p.zPos;
            xVel = p.xVel;
            yVel = p.yVel;
            zVel = p.zVel;

            return *this;
        }
    };

    template<concepts::MdSpan<BaseType> T_ViewType>
    struct ParticleData
    {
        T_ViewType masses;
        T_ViewType xPositions;
        T_ViewType yPositions;
        T_ViewType zPositions;
        T_ViewType xVelocities;
        T_ViewType yVelocities;
        T_ViewType zVelocities;

        constexpr ParticleData(
            T_ViewType const& masses,
            T_ViewType const& xPositions,
            T_ViewType const& yPositions,
            T_ViewType const& zPositions,
            T_ViewType const& xVelocities,
            T_ViewType const& yVelocities,
            T_ViewType const& zVelocities)
            : masses(masses)
            , xPositions(xPositions)
            , yPositions(yPositions)
            , zPositions(zPositions)
            , xVelocities(xVelocities)
            , yVelocities(yVelocities)
            , zVelocities(zVelocities)
        {
        }

        constexpr ParticleData(ParticleData const& rhs) = default;
        constexpr ParticleData(ParticleData&& rhs) = default;
        constexpr ParticleData& operator=(ParticleData const& rhs) = default;
        constexpr ParticleData& operator=(ParticleData&& rhs) = default;

        constexpr Particle operator[](IdxType idx) const&
        {
            return Particle(
                masses[idx],
                xPositions[idx],
                yPositions[idx],
                zPositions[idx],
                xVelocities[idx],
                yVelocities[idx],
                zVelocities[idx]);
        }

        constexpr RefParticle operator[](IdxType idx) &
        {
            return RefParticle(
                masses[idx],
                xPositions[idx],
                yPositions[idx],
                zPositions[idx],
                xVelocities[idx],
                yVelocities[idx],
                zVelocities[idx]);
        }
    };

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

template<alpaka::concepts::MdSpan<alpaka::example::nBody::BaseType> T_ViewType>
struct alpaka::trait::IsKernelArgumentTriviallyCopyable<alpaka::example::nBody::ParticleData<T_ViewType>>
    : std::true_type
{
};
