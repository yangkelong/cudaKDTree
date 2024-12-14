// ======================================================================== //
// Copyright 2022-2023 Ingo Wald                                            //
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

/* traversal with 'closest-corner-tracking' - somewhat better for some
   input distributions, by tracking the (N-dimensional) closest point
   in the given subtree's domain, rather than just always comparing
   only to the 1-dimensoinal plane */
#pragma once

namespace cukd {
  
  template<typename result_t, typename data_t,
           typename data_traits=default_data_traits<data_t>>
  inline __device__
  void traverse_cct(result_t &result,
                    typename data_traits::point_t queryPoint,
                    const box_t<typename data_traits::point_t> d_bounds,
                    const data_t *d_nodes,
                    int numPoints, int &traverse_node_num){
    using point_t = typename data_traits::point_t;
    using point_traits = ::cukd::point_traits<point_t>;
    using scalar_t = typename point_traits::scalar_t;
    enum { num_dims  = point_traits::num_dims };
      
    scalar_t cullDist = result.initialCullDist2();

    struct StackEntry {
      int nodeID;
      point_t closestCorner;
    };

    /* can do at most 2**30 points... */
    // 深度优先遍历
    StackEntry  stackBase[30];
    StackEntry *stackPtr = stackBase;

    int nodeID = 0;
    point_t closestPointOnSubtreeBounds = project(d_bounds, queryPoint);
    if (sqrDistance(queryPoint,closestPointOnSubtreeBounds) > cullDist)
      return;

    while (true) {
      if (nodeID >= numPoints) {  // 如果 nodeID 不合理, 从栈中取出一个 nodeID
        while (true) {
          if (stackPtr == stackBase)
            return;
          --stackPtr;
          closestPointOnSubtreeBounds = stackPtr->closestCorner;
          if (sqrDistance(closestPointOnSubtreeBounds, queryPoint) >= cullDist)
            continue;
          nodeID = stackPtr->nodeID;
          break;
        }
      }
      const auto &node = d_nodes[nodeID];  // 取出节点
      CUKD_STATS(if (cukd::g_traversalStats) ::atomicAdd(cukd::g_traversalStats, 1));
      traverse_node_num += 1;
      const point_t nodePoint = data_traits::get_point(node);
      {
        const auto sqrDist = sqrDistance(nodePoint, queryPoint);  // 计算当前节点到查询点距离平方
        cullDist = result.processCandidate(nodeID, sqrDist);  // 与当前最好记录比较
      }
      
      const int  dim
        = data_traits::has_explicit_dim
        ? data_traits::get_dim(d_nodes[nodeID])
        : (BinaryTree::levelOf(nodeID) % num_dims);  // 如果没有 has_explicit_dim, 采用轮流轴作划分维方向
      const auto node_dim   = get_coord(nodePoint, dim);
      const auto query_dim  = get_coord(queryPoint, dim);
      const bool  leftIsClose = query_dim < node_dim;  // 查询点位于左分支？
      const int   lChild = 2*nodeID+1;
      const int   rChild = lChild+1;

      auto farSideCorner = closestPointOnSubtreeBounds;
      const int farChild = leftIsClose ? rChild : lChild;
      point_traits::set_coord(farSideCorner, dim, node_dim);
      if (farChild < numPoints && sqrDistance(farSideCorner, queryPoint) < cullDist) {
        stackPtr->closestCorner = farSideCorner;
        stackPtr->nodeID = farChild;
        stackPtr++;
      }
      nodeID = leftIsClose ? lChild : rChild;
    }
  }


}
