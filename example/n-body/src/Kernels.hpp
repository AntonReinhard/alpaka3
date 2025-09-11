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
            for(concepts::Dim<2_idx> auto blockStartIdx : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::blocksInGrid,
                    IdxRange{Vec{0_idx, 0_idx}, Vec{extents.x(), extents.x()}, chunkSize}))
            {
                // blockStartIdx is now 2-dimensional, each block calculates the interactions between the x particles
                // and the y particles

                auto sharedParticlesX
                    = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, CVec<IdxType, chunkSize.y()>{});
                auto sharedParticlesY
                    = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, CVec<IdxType, chunkSize.y()>{});
                auto sharedParticlesZ
                    = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, CVec<IdxType, chunkSize.y()>{});

                auto otherParticlesX
                    = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, CVec<IdxType, chunkSize.x()>{});
                auto otherParticlesY
                    = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, CVec<IdxType, chunkSize.x()>{});
                auto otherParticlesZ
                    = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, CVec<IdxType, chunkSize.x()>{});
                auto otherParticlesMass
                    = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, CVec<IdxType, chunkSize.x()>{});

                auto accelX = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, CVec<IdxType, chunkSize.y()>{});
                auto accelY = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, CVec<IdxType, chunkSize.y()>{});
                auto accelZ = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, CVec<IdxType, chunkSize.y()>{});

                // == load the other particles (along x) into shared memory ==
                for(concepts::Dim<2_idx> auto particleIdx :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{1_idx, chunkSize.x()}}))
                {
                    otherParticlesX[particleIdx.x()] = particleData.xPositions[blockStartIdx.x() + particleIdx.x()];
                    otherParticlesY[particleIdx.x()] = particleData.yPositions[blockStartIdx.x() + particleIdx.x()];
                    otherParticlesZ[particleIdx.x()] = particleData.zPositions[blockStartIdx.x() + particleIdx.x()];
                    otherParticlesMass[particleIdx.x()] = particleData.masses[blockStartIdx.x() + particleIdx.x()];
                }

                // == load the shared particles (along y) into shared memory ==
                for(concepts::Dim<2_idx> auto particleIdx :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{chunkSize.y(), 1_idx}}))
                {
                    sharedParticlesX[particleIdx.y()] = particleData.xPositions[blockStartIdx.y() + particleIdx.y()];
                    sharedParticlesY[particleIdx.y()] = particleData.yPositions[blockStartIdx.y() + particleIdx.y()];
                    sharedParticlesZ[particleIdx.y()] = particleData.zPositions[blockStartIdx.y() + particleIdx.y()];
                    accelX[particleIdx.y()] = static_cast<BaseType>(0.);
                    accelY[particleIdx.y()] = static_cast<BaseType>(0.);
                    accelZ[particleIdx.y()] = static_cast<BaseType>(0.);
                }

                onAcc::syncBlockThreads(acc);

                // == iterate through every x,y pair of particles in this chunk ==
                for(concepts::Dim<2_idx> auto particleIdx :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkSize}))
                {
                    BaseType const r_x
                        = otherParticlesX[Vec{particleIdx.x()}] - sharedParticlesX[Vec{particleIdx.y()}];
                    BaseType const r_y
                        = otherParticlesY[Vec{particleIdx.x()}] - sharedParticlesY[Vec{particleIdx.y()}];
                    BaseType const r_z
                        = otherParticlesZ[Vec{particleIdx.x()}] - sharedParticlesZ[Vec{particleIdx.y()}];

                    BaseType const distSqr = r_x * r_x + r_y * r_y + r_z * r_z + EPS;
                    BaseType const distSixth = distSqr * distSqr * distSqr;
                    BaseType const invDistCube = 1.0f / math::sqrt(distSixth);
                    BaseType const a = otherParticlesMass[Vec{particleIdx.x()}] * invDistCube; // acceleration

                    accelX[particleIdx.y()] += a * r_x;
                    accelY[particleIdx.y()] += a * r_y;
                    accelZ[particleIdx.y()] += a * r_z;
                }

                onAcc::syncBlockThreads(acc);

                // == calculate and save the particles' new velocities back into global memory (along y) ==
                for(concepts::Dim<2_idx> auto particleIdx :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{chunkSize.y(), 1_idx}}))
                {
                    particleData.xVelocities[Vec{blockStartIdx.y() + particleIdx.y()}] += accelX[particleIdx.y()] * dt;
                    particleData.yVelocities[Vec{blockStartIdx.y() + particleIdx.y()}] += accelY[particleIdx.y()] * dt;
                    particleData.zVelocities[Vec{blockStartIdx.y() + particleIdx.y()}] += accelZ[particleIdx.y()] * dt;
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
