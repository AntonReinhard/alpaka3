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

            auto const constParticleData = particleData;

            // loop through blocks in the grid
            for(concepts::Dim<2_idx> auto blockStartIdx : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::blocksInGrid,
                    IdxRange{Vec{0_idx, 0_idx}, Vec{extents.x(), extents.x()}, chunkSize}))
            {
                // blockStartIdx is now 2-dimensional, each block calculates the interactions between the x particles
                // and the y particles

                auto sharedParticles
                    = onAcc::declareSharedMdArray<Particle, uniqueId()>(acc, CVec<IdxType, chunkSize.y()>{});
                auto otherParticles
                    = onAcc::declareSharedMdArray<Particle, uniqueId()>(acc, CVec<IdxType, chunkSize.x()>{});

                // == load the other particles (along x) into shared memory ==
                for(concepts::Dim<2_idx> auto particleIdx :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{1_idx, chunkSize.x()}}))
                {
                    otherParticles[particleIdx] = constParticleData[blockStartIdx.x() + particleIdx.x()];
                }

                // == load the shared particles (along y) into shared memory ==
                for(concepts::Dim<2_idx> auto particleIdx :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{chunkSize.y(), 1_idx}}))
                {
                    sharedParticles[particleIdx] = constParticleData[blockStartIdx.y() + particleIdx.y()];
                }

                // == iterate through every x,y pair of particles in this chunk ==
                for(concepts::Dim<2_idx> auto particleIdx :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkSize}))
                {
                    // x is the fast moving index, so use that for the "other" particles
                    sharedParticles[Vec{particleIdx.y()}].update(otherParticles[Vec{particleIdx.x()}], dt);
                }

                // == save the particles' updated velocities back into global memory (along y) ==
                for(concepts::Dim<2_idx> auto particleIdx :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{chunkSize.y(), 1_idx}}))
                {
                    particleData[blockStartIdx.y() + particleIdx.x()].xVel = sharedParticles[particleIdx].xVel;
                    particleData[blockStartIdx.y() + particleIdx.x()].yVel = sharedParticles[particleIdx].yVel;
                    particleData[blockStartIdx.y() + particleIdx.x()].zVel = sharedParticles[particleIdx].zVel;
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
            using namespace alpaka;

            for(concepts::Dim<1_idx> auto idx :
                onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{particleData.getExtents()}))
            {
                particleData[idx.x()].move(dt);
            }
        }
    };
} // namespace alpaka::example::nBody
