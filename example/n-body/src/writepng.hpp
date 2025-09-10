/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "particles.hpp"

#include <pngwriter.h>

#include <cmath>
#include <cstdlib>
#include <vector>

namespace alpaka::example::nBody
{
    template<typename T_View>
    void writePng(
        ParticleData<T_View> const& particles,
        concepts::MdSpan<Color> auto const& colors,
        std::string const& filename)
    {
        // Image dimensions
        constexpr int width = 1000;
        constexpr int height = 1000;

        // Create a black image
        pngwriter image(width, height, 0.0, filename.c_str());
        image.setcompressionlevel(9);

        // Project 3D particles to 2D by ignoring z coordinate
        for(IdxType idx = 0; idx < particles.getExtents().x(); ++idx)
        {
            auto const& p = particles[idx];

            // Ignore z coordinate, initial coordinates are between -1000 and 1000
            auto const px = (p.xPos + 1500) / 3000 * width;
            auto const py = (p.yPos + 1500) / 3000 * height;

            // Scale dot size inversely with distance
            auto const size = std::max(2.f, sqrtf(p.mass / 1e6f));
            int const size_ceil = ceill(size);

            auto const& c = colors[idx];
            // Draw a dot
            for(int dx = -size_ceil; dx <= size_ceil; ++dx)
            {
                for(int dy = -size_ceil; dy <= size_ceil; ++dy)
                {
                    int const x = roundl(px + dx);
                    int const y = roundl(py + dy);
                    if(x > 0 && x <= width && y > 0 && y <= height && sqrt(dx * dx + dy * dy) <= size)
                    {
                        image.plot(x, y, c.r, c.g, c.b);
                    }
                }
            }
        }

        // Save the image
        image.close();
    }
} // namespace alpaka::example::nBody
