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

#include "cukd/builder.h"
#include "cukd/fcp.h"  // fcp = "find closest point" query
#include <queue>
#include <iomanip>
#include <random>
#include <numeric>


using mydata3 = float3;
//using mydata3 = double3;
using mydata = typename cukd::scalar_type_of<mydata3>::type;


struct MyPoint { 
  mydata3  position;
  // 1 byte for split dimension
  uint8_t split_dim; 
};

struct MyPoint_traits : public cukd::default_data_traits<mydata3>{
  using point_t = mydata3;

  static inline __device__ __host__
  mydata3 get_point(const MyPoint &data)
  { return data.position; }

  static inline __device__ __host__
  mydata  get_coord(const MyPoint &data, int dim)
  { return cukd::get_coord(get_point(data),dim); }

  enum { has_explicit_dim = true };

  static inline __device__ void set_dim(MyPoint &p, int dim){p.split_dim = dim; }

  static inline __device__ int  get_dim(const MyPoint &p) { return p.split_dim; }
};

__global__ void d_fcp_mypoint(mydata *d_results, mydata3 *d_queries, int numQueries,
                      const cukd::box_t<mydata3> *d_bounds, MyPoint *d_nodes,
                      int numNodes, mydata cutOffRadius, int *d_records){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numQueries)
    return;
  mydata3 queryPos = d_queries[tid];  // 查询点 坐标
  cukd::FcpSearchParams params;
  params.cutOffRadius = cutOffRadius;
  int traverse_node_num = 0;
  int closestID = cukd::cct::fcp<MyPoint, MyPoint_traits>(queryPos, *d_bounds, d_nodes, numNodes, params, d_records[tid]);
  // d_records[tid] = traverse_node_num;
  d_results[tid] = (closestID < 0)
                       ? INFINITY
                       : cukd::distance(queryPos, d_nodes[closestID].position);
}




MyPoint *generateMyPoints(int N){
  static int g_seed = 100000;
  std::seed_seq seq{g_seed++};
  std::default_random_engine rd(seq);
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> int_dist(0, N);
  std::uniform_real_distribution<> double_dist(-1, 1);
  std::uniform_real_distribution<> double_dist_1(-10, 10);
  std::uniform_real_distribution<> double_dist_2(-5, 5);
  std::cout << "generating " << N << " uniform random points" << std::endl;
  MyPoint *d_points = 0;
  cudaMallocManaged((char **)&d_points, N * sizeof(*d_points));
  auto &dist = int_dist;
  if (!d_points)
    throw std::runtime_error("could not allocate points mem...");

  for (int i = 0; i < N; i++){
    d_points[i].position.x = (mydata)(dist(gen)/10000);
    d_points[i].position.y = (mydata)double_dist_1(gen);
    d_points[i].position.z = 0.;
  }
  return d_points;
}



template<typename T>
T *generatePoints(int N){
  static int g_seed = 100000;
  std::seed_seq seq{g_seed++};
  std::default_random_engine rd(seq);
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> int_dist(0, N);
  std::uniform_real_distribution<> double_dist(-1, 1);
  std::uniform_real_distribution<> double_dist_1(-10, 10);
  std::uniform_real_distribution<> double_dist_2(-5, 5);
  std::cout << "generating " << N << " uniform random points" << std::endl;
  T *d_points = 0;
  cudaMallocManaged((char **)&d_points, N * sizeof(*d_points));
  auto &dist = int_dist;
  if (!d_points)
    throw std::runtime_error("could not allocate points mem...");

  for (int i = 0; i < N; i++){
    d_points[i].x = (mydata)(dist(gen)/10000);
    d_points[i].y = (mydata)double_dist_1(gen);
    d_points[i].z = 0.;
    // d_points[i].z = (mydata)double_dist_2(gen);
  }
  return d_points;
}


__global__ void d_fcp_stackBased(mydata *d_results, mydata3 *d_queries, int numQueries,
                      /*! the world bounding box computed by the builder */
                      const cukd::box_t<mydata3> *d_bounds, mydata3 *d_nodes,
                      int numNodes, mydata cutOffRadius, int *d_records){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numQueries)
    return;
  mydata3 queryPos = d_queries[tid];  // 查询点 坐标
  cukd::FcpSearchParams params;
  params.cutOffRadius = cutOffRadius;
  int traverse_node_num = 0;
  int closestID = cukd::stackBased::fcp(queryPos, d_nodes, numNodes, params, d_records[tid]);
  // d_records[tid] = traverse_node_num;
  d_results[tid] = (closestID < 0)
                       ? INFINITY
                       : cukd::distance(queryPos, d_nodes[closestID]);
}

__global__ void d_fcp(mydata *d_results, mydata3 *d_queries, int numQueries,
                      /*! the world bounding box computed by the builder */
                      const cukd::box_t<mydata3> *d_bounds, mydata3 *d_nodes,
                      int numNodes, mydata cutOffRadius, int *d_records){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numQueries)
    return;
  mydata3 queryPos = d_queries[tid];  // 查询点 坐标
  cukd::FcpSearchParams params;
  params.cutOffRadius = cutOffRadius;
  int traverse_node_num = 0;
  int closestID = cukd::cct::fcp(queryPos, *d_bounds, d_nodes, numNodes, params, d_records[tid]);
  // d_records[tid] = traverse_node_num;
  d_results[tid] = (closestID < 0)
                       ? INFINITY
                       : cukd::distance(queryPos, d_nodes[closestID]);
}

int main(int ac, const char **av){
  // using namespace cukd::common;

  int numPoints = 1000000;
  int nRepeats = 1;
  size_t numQueries = 1000000;
  // 搜索时的最大半径
  mydata cutOffRadius = std::numeric_limits<mydata>::infinity();

  for (int i = 1; i < ac; i++)
  {
    std::string arg = av[i];
    if (arg[0] != '-')
      numPoints = std::stoi(arg);
    else if (arg == "-nq")
      numQueries = atoi(av[++i]);
    else if (arg == "-nr")
      nRepeats = atoi(av[++i]);
    else if (arg == "-r")
      cutOffRadius = std::stof(av[++i]);
    else
      throw std::runtime_error("known cmdline arg" + arg);
  }

  // ==================================================================
  // create sample input point that we'll build the tree over
  // ==================================================================


  mydata3 *d_points = generatePoints<mydata3>(numPoints);

  // ==================================================================
  // allocate some memory for the world-space bounding box, so the
  // builder can compute and return that for our chosen traversal
  // method to use
  // ==================================================================
  cukd::box_t<mydata3> *d_bounds;
  cudaMallocManaged((void **)&d_bounds, sizeof(cukd::box_t<mydata3>));
  std::cout << "allocated memory for the world space bounding box ..." << std::endl;

  // ==================================================================
  // build the tree. this will also comptue the world-space boudig box
  // of all points
  // ==================================================================
  std::cout << "calling builder..." << std::endl;
  double t0 = cukd::common::getCurrentTime();
  cukd::buildTree(d_points, numPoints, d_bounds);
  CUKD_CUDA_SYNC_CHECK();
  double t1 = cukd::common::getCurrentTime();
  std::cout << "done building tree, took "
            << cukd::common::prettyDouble(t1 - t0) << "s" << std::endl;

  // ==================================================================
  // create set of sample query points
  // ==================================================================
  mydata3 *d_queries = generatePoints<mydata3>(numQueries);
  // allocate memory for the results
  mydata *d_results;
  CUKD_CUDA_CALL(MallocManaged((void **)&d_results, numQueries * sizeof(*d_results)));
  // 记录每个查询点 遍历 tree 时 访问的节点数目
  int *d_records;
  cudaMallocManaged((char **)&d_records, numQueries * sizeof(int));
  // ==================================================================
  // and do some queryies - let's do the same ones in a loop so we cna
  // measure perf.
  // ==================================================================
  {
    for (int i = 0; i < numQueries; i++){
      d_records[i] = 0;
    }
    double t0 = cukd::common::getCurrentTime();
    for (int i = 0; i < nRepeats; i++)
    {
      int bs = 128;
      int nb = cukd::divRoundUp((int)numQueries, bs);
      d_fcp<<<nb, bs>>>(d_results, d_queries, numQueries, d_bounds, d_points, numPoints, cutOffRadius, d_records);
      cudaDeviceSynchronize();
    }
    CUKD_CUDA_SYNC_CHECK();
    double t1 = cukd::common::getCurrentTime();
    std::cout << "done " << nRepeats
              << " iterations of " << numQueries
              << " fcp queries, took " << cukd::common::prettyDouble(t1 - t0)
              << "s" << std::endl;
    std::cout << "that is " << cukd::common::prettyDouble(numQueries * nRepeats / (t1 - t0))
              << " queries/s" << std::endl;
    double avg_per_query = std::accumulate(d_records, d_records+numQueries, 0.)/numQueries;
    std::cout << "average traverse_node_num per query: " << avg_per_query << std::endl; 
  }

  {
    for (int i = 0; i < numQueries; i++){
      d_records[i] = 0;
    }
    double t0 = cukd::common::getCurrentTime();
    for (int i = 0; i < nRepeats; i++)
    {
      int bs = 128;
      int nb = cukd::divRoundUp((int)numQueries, bs);
      d_fcp_stackBased<<<nb, bs>>>(d_results, d_queries, numQueries, d_bounds, d_points, numPoints, cutOffRadius, d_records);
      cudaDeviceSynchronize();
    }
    CUKD_CUDA_SYNC_CHECK();
    double t1 = cukd::common::getCurrentTime();
    std::cout << "done " << nRepeats
              << " iterations of " << numQueries
              << " fcp queries, took " << cukd::common::prettyDouble(t1 - t0)
              << "s" << std::endl;
    std::cout << "that is " << cukd::common::prettyDouble(numQueries * nRepeats / (t1 - t0))
              << " queries/s" << std::endl;
    double avg_per_query = std::accumulate(d_records, d_records+numQueries, 0.)/numQueries;
    std::cout << "average traverse_node_num per query: " << avg_per_query << std::endl; 
  }


  { 
    //
    cukd::box_t<mydata3> *d_bounds;
    cudaMallocManaged((void **)&d_bounds, sizeof(cukd::box_t<mydata3>));
    MyPoint *d_points = generateMyPoints(numPoints);
    cukd::buildTree<MyPoint, MyPoint_traits>(d_points, numPoints, d_bounds);
    for (int i = 0; i < numQueries; i++){
      d_records[i] = 0;
    }
    double t0 = cukd::common::getCurrentTime();
    for (int i = 0; i < nRepeats; i++)
    {
      int bs = 128;
      int nb = cukd::divRoundUp((int)numQueries, bs);
      d_fcp_mypoint<<<nb, bs>>>(d_results, d_queries, numQueries, d_bounds, d_points, numPoints, cutOffRadius, d_records);
      cudaDeviceSynchronize();
    }
    CUKD_CUDA_SYNC_CHECK();
    double t1 = cukd::common::getCurrentTime();
    std::cout << "done " << nRepeats
              << " iterations of " << numQueries
              << " fcp queries, took " << cukd::common::prettyDouble(t1 - t0)
              << "s" << std::endl;
    std::cout << "that is " << cukd::common::prettyDouble(numQueries * nRepeats / (t1 - t0))
              << " queries/s" << std::endl;
    double avg_per_query = std::accumulate(d_records, d_records+numQueries, 0.)/numQueries;
    std::cout << "average traverse_node_num per query: " << avg_per_query << std::endl; 
  }
  
}
