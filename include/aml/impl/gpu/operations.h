#pragma once

#include <aml/array.h>
#include <aml/defs.h>
#include <aml/immutable_array.h>
#include <aml/shape.h>

namespace aml {
namespace impl {
namespace gpu {

/** SET ***********************************************************************/

template <typename T, int Dim>
void set(Array<T, Dim> &out, const T &val) {
  // TODO
}

/** UNARY_OP ******************************************************************/

template <typename Tin, typename Tout, int Dim, typename Op>
void unary_op(const ImmutableArray<Tin, Dim> &in,
              Array<Tout, Dim> &out,
              const Op &op) {
  // TODO
}

/** BINARY_OP *****************************************************************/


template <typename Tin1, typename Tin2, typename Tout, int Dim, typename Op>
void binary_op(const ImmutableArray<Tin1, Dim> &in1,
               const ImmutableArray<Tin2, Dim> &in2,
               Array<Tout, Dim> &out,
               const Op &op) {
  // TODO
}

/** REDUCE ********************************************************************/

template <typename Tin,
          int DimIn,
          typename Tout,
          int DimOut,
          typename TransformOp,
          typename ReduceOp>
void reduce(const ImmutableArray<Tin, DimIn> &in,
            Array<Tout, DimOut> &out,
            const TransformOp &op_t,
            const ReduceOp &op_r,
            Shape<DimIn> stride) {
  // TODO
}

}  // namespace gpu
}  // namespace impl
}  // namespace aml

