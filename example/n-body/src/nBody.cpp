/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#include "UpdateKernel.hpp"
#include "alpaka/onHost/FrameSpec.hpp"
#include "common.hpp"
#include "helpers.hpp"
#include "particles.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
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

        using IdxTypeVec = Vec<IdxType, 1u>;

        std::cout << "\nRunning accelerator: " << std::endl;
        std::cout << deviceSpec.getApi().getName() << std::endl;

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

        std::cout << "Initializing data on host" << std::endl;
        initMasses(massesHost);
        initPositions(xPositionsHost, yPositionsHost, zPositionsHost);
        initVelocities(xVelocitiesHost, yVelocitiesHost, zVelocitiesHost);
        std::cout << "Done initializing data on host" << std::endl;

        auto massesDev = onHost::allocLike(devAcc, massesHost);
        auto xPositionsDev = onHost::allocLike(devAcc, xPositionsHost);
        auto yPositionsDev = onHost::allocLike(devAcc, yPositionsHost);
        auto zPositionsDev = onHost::allocLike(devAcc, zPositionsHost);
        auto xVelocitiesDev = onHost::allocLike(devAcc, xVelocitiesHost);
        auto yVelocitiesDev = onHost::allocLike(devAcc, yVelocitiesHost);
        auto zVelocitiesDev = onHost::allocLike(devAcc, zVelocitiesHost);

        auto massesDevDouble = onHost::allocLike(devAcc, massesHost);
        auto xPositionsDevDouble = onHost::allocLike(devAcc, xPositionsHost);
        auto yPositionsDevDouble = onHost::allocLike(devAcc, yPositionsHost);
        auto zPositionsDevDouble = onHost::allocLike(devAcc, zPositionsHost);
        auto xVelocitiesDevDouble = onHost::allocLike(devAcc, xVelocitiesHost);
        auto yVelocitiesDevDouble = onHost::allocLike(devAcc, yVelocitiesHost);
        auto zVelocitiesDevDouble = onHost::allocLike(devAcc, zVelocitiesHost);

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

        auto particleData = ParticleData(
            massesDev,
            xPositionsDev,
            yPositionsDev,
            zPositionsDev,
            xVelocitiesDev,
            yVelocitiesDev,
            zVelocitiesDev);

        // this does not have to be initialized because it will be overridden by the first simulation step anyway
        auto particleDataDoubleBuf = ParticleData(
            massesDevDouble,
            xPositionsDevDouble,
            yPositionsDevDouble,
            zPositionsDevDouble,
            xVelocitiesDevDouble,
            yVelocitiesDevDouble,
            zVelocitiesDevDouble);

        // Appropriate chunk size to split your problem for your Acc
        constexpr auto chunkSize = CVec<IdxType, 256u>{};

        IdxTypeVec const numChunks{
            divCeil(extents[0], chunkSize[0]),
        };

        // Make an instance of the kernel object to use later
        UpdateKernel updateKernel;

        // The frame spec describes the size of blocks and the grid used on the accelerator.
        auto const frameSpec = FrameSpec{numChunks, chunkSize};

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
                KernelBundle{updateKernel, particleData, particleDataDoubleBuf, dt});

            // We just swap next and curr (shallow copy)
            std::swap(particleDataDoubleBuf, particleData);
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
        std::cerr << "  -n numParticles: Number of particles. Default: 2^10 = 1024" << std::endl;
        std::cerr << "  -t numTimeSteps: Number of time steps that the simulation is run for. Default: 100"
                  << std::endl;
        std::cerr << "  -d dt: Delta t for the timesteps. Default: 0.1" << std::endl;
        std::cerr << "  -h: Print this help message" << std::endl;
        std::cerr << std::endl;
    }

} // namespace alpaka::example::nBody

auto main(int argc, char* argv[]) -> int
{
    using namespace alpaka;
    using namespace alpaka::example::nBody;

    // Default value if no command line argument used
    IdxType numParticles = 1 << 10;
    IdxType numTimeSteps = 100;

    int opt;
    BaseType dt = 0.1;

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
