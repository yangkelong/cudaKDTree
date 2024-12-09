// ======================================================================== //
// Copyright 2018-2023 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //
#pragma once
#include "cukd/builder.h"
#include "cukd/fcp.h"  // fcp = "find closest point" query
#include <queue>
#include <iomanip>

//using mydata3 = float3;
//using mydata = float;
using mydata3 = double3;
using mydata = double;

template<typename T>
T* uploadPoints(mydata* coords, int N) {
    T* d_points = 0;
    cudaMallocManaged((char**)&d_points, N * sizeof(*d_points));
    if (!d_points)
        throw std::runtime_error("could not allocate points mem...");
    for (int i = 0; i < N; i++) {
        d_points[i].x = (mydata)coords[3 * i];
        d_points[i].y = (mydata)coords[3 * i + 1];
        d_points[i].z = (mydata)coords[3 * i + 2];
    }
    return d_points;
}

// 查询host函数, cct
extern "C"
__global__ void d_fcp(mydata* d_results,
    mydata3* d_queries,
    int numQueries,
    /*! the world bounding box computed by the builder */
    const cukd::box_t<mydata3>* d_bounds,
    mydata3* d_nodes,
    int numNodes,
    mydata cutOffRadius) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numQueries)
        return;

    mydata3 queryPos = d_queries[tid];
    cukd::FcpSearchParams params;
    params.cutOffRadius = cutOffRadius;
    int closestID = cukd::cct::fcp(queryPos, *d_bounds, d_nodes, numNodes, params);

    d_results[tid] = (closestID < 0)
        ? INFINITY
        : cukd::distance(queryPos, d_nodes[closestID]);
}


// 后续考虑将构建 查询 分开？
void calWalDist();
extern "C" __host__ void
toWallDistance(mydata* coords, unsigned int n, mydata* query_coords, unsigned int numQueries,
    mydata* result_coords) {
    using namespace cukd::common;
    unsigned int numPoints;
    cukd::box_t<mydata3>* d_bounds;
    mydata3* d_points = uploadPoints<mydata3>(coords, n);
    cudaMallocManaged((void**)&d_bounds, sizeof(cukd::box_t<mydata3>));
    std::cout << "allocated memory for the world space bounding box ..." << std::endl;
    // ==================================================================
    // build the tree.
    // ==================================================================
    std::cout << "calling builder..." << std::endl;
    double t0 = cukd::common::getCurrentTime();
    cukd::buildTree(d_points, numPoints, d_bounds);
    CUKD_CUDA_SYNC_CHECK();
    double t1 = cukd::common::getCurrentTime();
    std::cout << "done building tree, took "
        << cukd::common::prettyDouble(t1 - t0) << "s" << std::endl;
    // 搜索时的最大半径
    mydata cutOffRadius = std::numeric_limits<mydata>::infinity();
    mydata3* d_queries = uploadPoints<mydata3>(coords, n);
    // allocate memory for the results
    mydata* d_results;
    CUKD_CUDA_CALL(MallocManaged((void**)&d_results, numQueries * sizeof(*d_results)));
    // ==================================================================
    // do queryies 
    // ==================================================================
    t0 = cukd::common::getCurrentTime();
    int bs = 128;
    int nb = cukd::divRoundUp((int)numQueries, bs);
    d_fcp << <nb, bs >> > (d_results, d_queries, numQueries,
        d_bounds, d_points, numPoints, cutOffRadius);
    cudaDeviceSynchronize();
    CUKD_CUDA_SYNC_CHECK();
    t1 = cukd::common::getCurrentTime();
    std::cout << "done "
        << " iterations of " << numQueries
        << " fcp queries, took " << cukd::common::prettyDouble(t1 - t0)
        << "s" << std::endl;
    std::cout << "that is " << cukd::common::prettyDouble(numQueries / (t1 - t0))
        << " queries/s" << std::endl;
    // 
    for (int i = 0; i < numQueries; i++) {
        result_coords[3 * i] = d_queries[i].x;
        result_coords[3 * i + 1] = d_queries[i].y;
        result_coords[3 * i + 2] = d_queries[i].z;
    }
    cudaFree(d_points);
    cudaFree(d_bounds);
    cudaFree(d_results);
    cudaFree(d_results);
}

