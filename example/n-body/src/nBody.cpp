/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#include "Kernels.hpp"
#include "alpaka/onHost/FrameSpec.hpp"
#include "common.hpp"
#include "helpers.hpp"
#include "particles.hpp"

#ifdef PNGWRITER_ENABLED
#    include "writepng.hpp"
#endif

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>

namespace alpaka::example::nBody
{
    void printExampleHeader(IdxType const numParticles, IdxType const numTimeSteps, BaseType const dt)
    {
        std::cout << "================================" << std::endl;
        std::cout << "Example n-Body simulation" << std::endl;
        std::cout << "    Number of particles [#]: " << numParticles << std::endl;
        std::cout << "    Time steps [#]: " << numTimeSteps << std::endl;
        std::cout << "    dt: " << dt << std::endl;
        std::cout << "================================" << std::endl;
        std::cout << std::endl;
    }

    /** @brief Run an n-body simulation.
     *
     * Some number of particles are simulated in 3-dimensional space. Each particle gets a mass, position vector, and a
     * velocity vector, initialized randomly (see common.hpp for constants to tweak the generation). The particles then
     * interact gravitationally. The UpdateVelocitiesKernel updates each particle's velocity for one timestep and
     * writes them back. The updated velocities only depend on the other particles' positions and masses, so there is
     * no data race. Next, the UpdatePositionsKernel moves each particle independently according to its current
     * velocity. This repeats for the desired number of time steps.
     *
     * @param deviceSpec The device specification to run on.
     * @param computeExec The device to execute on.
     * @param numParticles The number of particles to simulate.
     * @param numTimeSteps The number of time steps to run for.
     * @param dt The delta t to use as time steps.
     */
    int example(
        auto const deviceSpec,
        auto const computeExec,
        IdxType const numParticles,
        IdxType const numTimeSteps,
        BaseType const dt)
    {
        using namespace alpaka;
        using namespace alpaka::onHost;

        using IdxTypeVec = Vec<IdxType, 1_idx>;

        std::cout << "\nRunning accelerator: " << deviceSpec.getApi().getName() << std::endl;

        auto devSelector = makeDeviceSelector(deviceSpec);
        Device devAcc = devSelector.makeDevice(0);

        // simulation defines
        // {Y, X}
        IdxTypeVec const extents = Vec{numParticles};

        auto massesHost = onHost::allocHost<BaseType>(extents);
        auto xPositionsHost = onHost::allocHost<BaseType>(extents);
        auto yPositionsHost = onHost::allocHost<BaseType>(extents);
        auto zPositionsHost = onHost::allocHost<BaseType>(extents);
        auto xVelocitiesHost = onHost::allocHost<BaseType>(extents);
        auto yVelocitiesHost = onHost::allocHost<BaseType>(extents);
        auto zVelocitiesHost = onHost::allocHost<BaseType>(extents);

        auto particleDataHost = ParticleData{
            massesHost.getView(),
            xPositionsHost.getView(),
            yPositionsHost.getView(),
            zPositionsHost.getView(),
            xVelocitiesHost.getView(),
            yVelocitiesHost.getView(),
            zVelocitiesHost.getView(),
        };

        std::cout << "Initializing data on host" << std::endl;
        initMasses(massesHost);
        initPositions(xPositionsHost, yPositionsHost, zPositionsHost);
        initVelocities(
            xVelocitiesHost,
            yVelocitiesHost,
            zVelocitiesHost,
            xPositionsHost,
            yPositionsHost,
            zPositionsHost);

#ifdef PNGWRITER_ENABLED
        auto colors = onHost::allocHost<Color>(extents);
        initColors(colors);
#endif

        std::cout << "Done initializing data on host" << std::endl;

        auto massesDev = onHost::allocLike(devAcc, massesHost);
        auto xPositionsDev = onHost::allocLike(devAcc, xPositionsHost);
        auto yPositionsDev = onHost::allocLike(devAcc, yPositionsHost);
        auto zPositionsDev = onHost::allocLike(devAcc, zPositionsHost);
        auto xVelocitiesDev = onHost::allocLike(devAcc, xVelocitiesHost);
        auto yVelocitiesDev = onHost::allocLike(devAcc, yVelocitiesHost);
        auto zVelocitiesDev = onHost::allocLike(devAcc, zVelocitiesHost);

        // Select queue
        Queue dumpQueue = devAcc.makeQueue();
        Queue computeQueue = devAcc.makeQueue();

        // copy data to device
        onHost::memcpy(dumpQueue, massesDev, massesHost);
        onHost::memcpy(dumpQueue, xPositionsDev, xPositionsHost);
        onHost::memcpy(dumpQueue, yPositionsDev, yPositionsHost);
        onHost::memcpy(dumpQueue, zPositionsDev, zPositionsHost);
        onHost::memcpy(dumpQueue, xVelocitiesDev, xVelocitiesHost);
        onHost::memcpy(dumpQueue, yVelocitiesDev, yVelocitiesHost);
        onHost::memcpy(dumpQueue, zVelocitiesDev, zVelocitiesHost);
        onHost::wait(dumpQueue);
        std::cout << "Data copied to device" << std::endl;

        auto particleData = ParticleData{
            massesDev.getView(),
            xPositionsDev.getView(),
            yPositionsDev.getView(),
            zPositionsDev.getView(),
            xVelocitiesDev.getView(),
            yVelocitiesDev.getView(),
            zVelocitiesDev.getView()};

        // Appropriate chunk size to split your problem for your Acc
        constexpr auto chunkSize = CVec<IdxType, 256_idx>{};

        auto const numChunks = divCeil(Vec{extents.x()}, chunkSize);

        // The frame spec describes the size of blocks and the grid used on the accelerator.
        auto const frameSpec = FrameSpec{numChunks, chunkSize};

        // Make an instance of the kernel object to use later
        UpdateVelocitiesKernel updateVelocitiesKernel;
        UpdatePositionsKernel updatePositionsKernel;

        auto const startTime = std::chrono::high_resolution_clock::now();

        // Simulate
        for(IdxType step = 1; step <= numTimeSteps; ++step)
        {
            // Queue one step of the simulation
            // The kernel bundle contains the kernel first, then all its arguments *except* the accelerator, which is
            // implicitly passed by alpaka.
            computeQueue.enqueue(
                computeExec,
                frameSpec,
                KernelBundle{updateVelocitiesKernel, particleData, chunkSize, dt});

            computeQueue.enqueue(computeExec, frameSpec, KernelBundle{updatePositionsKernel, particleData, dt});


#ifdef PNGWRITER_ENABLED
            if(step % pngStepSize == 0)
            {
                wait(computeQueue);

                // copy data to host
                onHost::memcpy(dumpQueue, massesHost, particleData.masses);
                onHost::memcpy(dumpQueue, xPositionsHost, particleData.xPositions);
                onHost::memcpy(dumpQueue, yPositionsHost, particleData.yPositions);
                onHost::memcpy(dumpQueue, zPositionsHost, particleData.zPositions);
                onHost::memcpy(dumpQueue, xVelocitiesHost, particleData.xVelocities);
                onHost::memcpy(dumpQueue, yVelocitiesHost, particleData.yVelocities);
                onHost::memcpy(dumpQueue, zVelocitiesHost, particleData.zVelocities);
                onHost::wait(dumpQueue);

                wait(dumpQueue);

                std::ostringstream oss;
                oss << std::setw(5) << std::setfill('0') << (step / pngStepSize);

                writePng(particleDataHost, colors, "particles_" + oss.str() + ".png");
            }
#endif
        }

        // Wait for the entire queue to be finished
        wait(computeQueue);
        auto const endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTime = endTime - startTime;

        std::cout << "The simulation took " << elapsedTime.count() << " seconds." << std::endl;
        std::cout << "Time per time step: " << elapsedTime.count() / numTimeSteps * 1000 << " ms." << std::endl;

        return EXIT_SUCCESS;
    }

    void help(char* argv[])
    {
        std::cerr << argv[0] << " [OPTIONS]" << std::endl;
        std::cerr << "  -n numParticles: Number of particles. Default: " << defaultNumParticles << std::endl;
        std::cerr << "  -t numTimeSteps: Number of time steps that the simulation is run for. Default: "
                  << defaultTimeSteps << std::endl;
        std::cerr << "  -d dt: Delta t for the timesteps. Default: " << defaultTimeSteps << std::endl;
        std::cerr << "  -h: Print this help message" << std::endl;
        std::cerr << std::endl;
    }

} // namespace alpaka::example::nBody

auto main(int argc, char* argv[]) -> int
{
    using namespace alpaka;
    using namespace alpaka::example::nBody;

    // Default value if no command line argument used
    IdxType numParticles = defaultNumParticles;
    IdxType numTimeSteps = defaultTimeSteps;
    BaseType dt = defaultDt;

    int opt;

    while((opt = getopt(argc, argv, "hn:t:d:")) != -1)
    {
        switch(opt)
        {
        case 'n':
            try
            {
                numParticles = static_cast<IdxType>(std::stoull(optarg, nullptr, 0));
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for unsigned long long.\n";
                return EXIT_FAILURE;
            }
            break;
        case 't':
            try
            {
                numTimeSteps = static_cast<IdxType>(std::stoull(optarg, nullptr, 0));
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for unsigned long long.\n";
                return EXIT_FAILURE;
            }
            break;
        case 'd':
            try
            {
                dt = std::stod(optarg, nullptr);
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for unsigned long long.\n";
                return EXIT_FAILURE;
            }
            break;
        case 'h':
            help(argv);
            exit(EXIT_SUCCESS);
        default:
            help(argv);
            exit(EXIT_FAILURE);
        }
    }

    printExampleHeader(numParticles, numTimeSteps, dt);

    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        {
            return alpaka::example::nBody::example(
                backend[object::deviceSpec],
                backend[object::exec],
                numParticles,
                numTimeSteps,
                dt);
        },
        onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors));
}
