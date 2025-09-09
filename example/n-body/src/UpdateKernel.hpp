/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "helpers.hpp"

#include <alpaka/alpaka.hpp>

namespace alpaka::example::nBody
{
    struct UpdateKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC auto operator()(
            T_Acc const& acc,
            auto const& particleData,
            auto particleDataDoubleBuf,
            concepts::Vector auto extents,
            BaseType const dt) const -> void
        {
            using namespace alpaka;

            for(auto particleIdx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{extents}))
            {
                Particle tempP = particleData[particleIdx.x()];

                for(auto otherParticleIdx = 0; otherParticleIdx < extents.x(); ++otherParticleIdx)
                {
                    // no need to skip self because softening factor prevents div by zero
                    tempP.update(particleData[otherParticleIdx]);
                }
                particleDataDoubleBuf[particleIdx.x()] = tempP;
            }
        }
    };
} // namespace alpaka::example::nBody
