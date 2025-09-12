/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "particles.hpp"

#include <alpaka/alpaka.hpp>

namespace alpaka::example::nBody
{
    /** @brief The main kernel, computing one step of the simulation for all particles.
     *
     * @param acc The accelerator the kernel runs on, automatically passed by alpaka
     * @param particleData The ParticleData of the current state of particles.
     * @param chunkSize The chunk size for the calculation (1D). Will also be used as shared memory size, so it has to
     * be a CVector (known at compile time).
     * @param dt The delta t time step to compute.
     */
    struct UpdateVelocitiesKernel
    {
        template<typename T_Acc, typename T_View>
        ALPAKA_FN_ACC auto operator()(
            T_Acc const& acc,
            ParticleData<T_View> particleData,
            concepts::CVector auto const chunkSize,
            BaseType const dt) const -> void
        {
            using namespace alpaka;

            auto const extents = particleData.getExtents();

            // loop through blocks in the grid
            for(concepts::Dim<1_idx> auto blockStartIdx : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::blocksInGrid,
                    IdxRange{CVec<IdxType, 0_idx>{}, extents, chunkSize}))
            {
                // each block updates its particles with all other particles
                auto sharedParticlesX = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, chunkSize);
                auto sharedParticlesY = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, chunkSize);
                auto sharedParticlesZ = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, chunkSize);

                auto accelX = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, chunkSize);
                auto accelY = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, chunkSize);
                auto accelZ = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, chunkSize);

                // the data of the other particles for each tile. will be overridden in the inner loop
                auto otherParticlesX = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, chunkSize);
                auto otherParticlesY = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, chunkSize);
                auto otherParticlesZ = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, chunkSize);
                auto otherParticlesMass = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, chunkSize);

                // == load the particles for this block into shared memory and initialize accelerations ==
                for(concepts::Dim<1_idx> auto particleIdx :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkSize}))
                {
                    sharedParticlesX[particleIdx] = particleData.xPositions[blockStartIdx + particleIdx];
                    sharedParticlesY[particleIdx] = particleData.yPositions[blockStartIdx + particleIdx];
                    sharedParticlesZ[particleIdx] = particleData.zPositions[blockStartIdx + particleIdx];
                    accelX[particleIdx] = static_cast<BaseType>(0.);
                    accelY[particleIdx] = static_cast<BaseType>(0.);
                    accelZ[particleIdx] = static_cast<BaseType>(0.);
                }

                // == loop through all other particles in chunks ==
                for(auto otherBlockStartIdx = 0_idx; otherBlockStartIdx < extents.x();
                    otherBlockStartIdx += chunkSize.x())
                {
                    // == load the particles for this block into shared memory and initialize accelerations ==
                    for(concepts::Dim<1_idx> auto particleIdx :
                        onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkSize}))
                    {
                        otherParticlesX[particleIdx] = particleData.xPositions[otherBlockStartIdx + particleIdx];
                        otherParticlesY[particleIdx] = particleData.yPositions[otherBlockStartIdx + particleIdx];
                        otherParticlesZ[particleIdx] = particleData.zPositions[otherBlockStartIdx + particleIdx];
                        otherParticlesMass[particleIdx] = particleData.masses[otherBlockStartIdx + particleIdx];
                    }

                    onAcc::syncBlockThreads(acc);

                    // == iterate through every x,y pair of particles in this tile ==
                    for(concepts::Dim<1_idx> auto particleIdx :
                        onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkSize}))
                    {
                        for(IdxType i = 0_idx; i < chunkSize; ++i)
                        {
                            BaseType const r_x = otherParticlesX[i] - sharedParticlesX[particleIdx];
                            BaseType const r_y = otherParticlesY[i] - sharedParticlesY[particleIdx];
                            BaseType const r_z = otherParticlesZ[i] - sharedParticlesZ[particleIdx];

                            BaseType const distSqr = r_x * r_x + r_y * r_y + r_z * r_z + EPS;
                            BaseType const distSixth = distSqr * distSqr * distSqr;
                            BaseType const invDistCube = 1.0f / math::sqrt(distSixth);
                            BaseType const a = otherParticlesMass[i] * invDistCube; // acceleration

                            accelX[particleIdx] += a * r_x;
                            accelY[particleIdx] += a * r_y;
                            accelZ[particleIdx] += a * r_z;
                        }
                    }

                    onAcc::syncBlockThreads(acc);
                }

                onAcc::syncBlockThreads(acc);

                // == calculate and save the particles' new velocities back into global memory ==
                for(concepts::Dim<1_idx> auto particleIdx :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkSize}))
                {
                    particleData.xVelocities[blockStartIdx + particleIdx] += accelX[particleIdx] * dt;
                    particleData.yVelocities[blockStartIdx + particleIdx] += accelY[particleIdx] * dt;
                    particleData.zVelocities[blockStartIdx + particleIdx] += accelZ[particleIdx] * dt;
                }
            }
        }
    };

    /** @brief The kernel computing each particle's updated position by moving it.
     *
     * @param acc The accelerator the kernel runs on, automatically passed by alpaka.
     * @param particleData The ParticleData of the current state of particles.
     * @param chunkSize The chunk size for the calculation (1D). Will also be used as shared memory size, so it has
     * to be a CVector (known at compile time).
     * @param dt The delta t time step to compute.
     */
    struct UpdatePositionsKernel
    {
        template<typename T_Acc, typename T_View>
        ALPAKA_FN_ACC auto operator()(T_Acc const& acc, ParticleData<T_View> particleData, BaseType const dt) const
            -> void
        {
            auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};

            simdGrid.concurrent(
                acc,
                particleData.getExtents(),
                [&](auto const&,
                    auto&& simdX,
                    auto&& simdY,
                    auto&& simdZ,
                    auto const&& simdXVel,
                    auto const&& simdYVel,
                    auto const&& simdZVel) constexpr
                {
                    auto x = simdX.load();
                    auto xV = simdXVel.load();
                    x += xV * dt;
                    simdX = x;
                    auto y = simdY.load();
                    auto yV = simdYVel.load();
                    y += yV * dt;
                    simdY = y;
                    auto z = simdZ.load();
                    auto zV = simdZVel.load();
                    z += zV * dt;
                    simdZ = z;
                },
                particleData.xPositions,
                particleData.yPositions,
                particleData.zPositions,
                particleData.xVelocities,
                particleData.yVelocities,
                particleData.zVelocities);
        }
    };
} // namespace alpaka::example::nBody
