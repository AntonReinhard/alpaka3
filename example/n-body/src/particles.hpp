/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "common.hpp"

#include <alpaka/alpaka.hpp>

namespace alpaka::example::nBody
{
    /** @brief Particle type with mass, 3-dimensional position, and 3-dimensional velocity.
     */
    struct Particle
    {
        BaseType mass;
        BaseType xPos;
        BaseType yPos;
        BaseType zPos;
        BaseType xVel;
        BaseType yVel;
        BaseType zVel;

        /** @brief Update the particle's velocities by interacting it with the given other particle.
         */
        constexpr void update(Particle const& other, BaseType const dt)
        {
            BaseType const r_x = other.xPos - xPos;
            BaseType const r_y = other.yPos - yPos;
            BaseType const r_z = other.zPos - zPos;

            BaseType const distSqr = r_x * r_x + r_y * r_y + r_z * r_z + EPS;
            BaseType const distSixth = distSqr * distSqr * distSqr;
            BaseType const invDistCube = 1.0f / math::sqrt(distSixth);
            BaseType const a = other.mass * invDistCube; // acceleration

            xVel += a * r_x * dt;
            yVel += a * r_y * dt;
            zVel += a * r_z * dt;
        }

        /** @brief Move the particle according to its current velocity and the given delta t.
         */
        constexpr void move(BaseType const dt)
        {
            xPos += xVel * dt;
            yPos += yVel * dt;
            zPos += zVel * dt;
        }
    };

    /** @brief A particle struct holding references of the particle fields (mass, positions, velocities). Used to set
     * particle values in ParticleData without copying all fields first.
     */
    struct RefParticle
    {
        BaseType& mass;
        BaseType& xPos;
        BaseType& yPos;
        BaseType& zPos;
        BaseType& xVel;
        BaseType& yVel;
        BaseType& zVel;

        /** @brief Set the referenced particle's fields to the given particle's fields.
         */
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

    /** @brief A struct of arrays holding masses, 3-dimensional positions, and 3-dimensional velocities of a number of
     * particles.
     *
     * @tparam T_View The type of view used for each of the arrays.
     */
    template<concepts::MdSpan<BaseType> T_View>
    struct ParticleData
    {
        T_View masses;
        T_View xPositions;
        T_View yPositions;
        T_View zPositions;
        T_View xVelocities;
        T_View yVelocities;
        T_View zVelocities;

        constexpr ParticleData(
            T_View const& masses,
            T_View const& xPositions,
            T_View const& yPositions,
            T_View const& zPositions,
            T_View const& xVelocities,
            T_View const& yVelocities,
            T_View const& zVelocities)
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
        ALPAKA_FN_HOST_ACC ~ParticleData() = default;

        constexpr auto getExtents() const&
        {
            return masses.getExtents();
        }

        /** @brief Constant access operator, returning a copied Particle.
         */
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

        /** @brief Ref access operator, returning a RefParticle without copying.
         */
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
} // namespace alpaka::example::nBody

// Implement this trait to allow using ParticleData as an alpaka kernel argument.
template<alpaka::concepts::MdSpan<alpaka::example::nBody::BaseType> T_ViewType>
struct alpaka::trait::IsKernelArgumentTriviallyCopyable<alpaka::example::nBody::ParticleData<T_ViewType>>
    : std::true_type
{
};
