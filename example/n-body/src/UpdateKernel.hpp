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
     * @param particleDataDoubleBuf The buffer where the next state of particles is written by this kernel.
     * @param dt The delta t time step to compute.
     */
    struct UpdateKernel
    {
        template<typename T_Acc, typename T_View>
        ALPAKA_FN_ACC auto operator()(
            T_Acc const& acc,
            ParticleData<T_View> const& particleData,
            ParticleData<T_View> particleDataDoubleBuf,
            BaseType const dt) const -> void
        {
            using namespace alpaka;

            auto const extents = particleData.getExtents();

            for(auto particleIdx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{extents}))
            {
                Particle tempP = particleData[particleIdx.x()];

                // calculate forces between all other particles and tempP, updating its velocity each time
                for(auto otherParticleIdx = 0; otherParticleIdx < extents.x(); ++otherParticleIdx)
                {
                    // no need to skip self because softening factor prevents div by zero and the force is zero anyway
                    tempP.update(particleData[otherParticleIdx], dt);
                }

                // finally, update the particle's positions
                tempP.move(dt);
                particleDataDoubleBuf[particleIdx.x()] = tempP;
            }
        }
    };
} // namespace alpaka::example::nBody
