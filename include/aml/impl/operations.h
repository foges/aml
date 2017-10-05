#pragma once

#include <cblas.h>

#include <aml/defs.h>

namespace aml {
namespace impl {
namespace cpu {

/** GEMM **********************************************************************/

inline CBLAS_TRANSPOSE convert_op(OP op) {
  return op == NO_TRANS ? CblasNoTrans : CblasTrans;
}

inline void gemm(OP op_a,
                 OP op_b,
                 Index m,
                 Index n,
                 Index k,
                 float alpha,
                 const float *a,
                 Index lda,
                 const float *b,
                 Index ldb,
                 float beta,
                 float *c,
                 Index ldc) {
  cblas_sgemm(CblasColMajor, convert_op(op_a), convert_op(op_b),
      m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void gemm(OP op_a,
                 OP op_b,
                 Index m,
                 Index n,
                 Index k,
                 double alpha,
                 const double *a,
                 Index lda,
                 const double *b,
                 Index ldb,
                 double beta,
                 double *c,
                 Index ldc) {
  cblas_dgemm(CblasColMajor, convert_op(op_a), convert_op(op_b),
      m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** SET ***********************************************************************/

template <typename T>
void set(T *out,
         const Shape<0> &stride,
         const Shape<0> &shape,
         const T &val) {
  *out = val;
}

template <typename T, int Dim>
void set(T *out,
         const Shape<Dim> &stride,
         const Shape<Dim> &shape,
         const T &val) {
  for (Index i = 0; i < shape.head(); ++i) {
    set(out + i * stride.head(), stride.tail(), shape.tail(), val);
  }
}

template <typename T, int Dim>
void set(Array<T, Dim> &out, const T &val) {
  set(out.data(), out.stride(), out.shape(), val);
}

/** UNARY_OP ******************************************************************/

template <typename Tin, typename Tout, typename Op>
void unary_op(const Tin *in,
              Shape<0> in_stride,
              Tout *out,
              Shape<0> out_stride,
              Shape<0> shape,
              const Op &op) {
  *out = op(*in);
}

template <typename Tin, typename Tout, int Dim, typename Op>
void unary_op(const Tin *in,
              Shape<Dim> in_stride,
              Tout *out,
              Shape<Dim> out_stride,
              Shape<Dim> shape,
              const Op &op) {
  for (Index i = 0; i < shape.head(); ++i) {
    unary_op(
        in + i * in_stride.head(), in_stride.tail(),
        out + i * out_stride.head(), out_stride.tail(), shape.tail(), op);
  }
}

template <typename Tin, typename Tout, int Dim, typename Op>
void unary_op(const ImmutableArray<Tin, Dim> &in,
              Array<Tout, Dim> &out,
              const Op &op) {
  unary_op(in.data(), in.stride(), out.data(), out.stride(), in.shape(), op);
}

/** BINARY_OP *****************************************************************/

template <typename Tin1, typename Tin2, typename Tout, typename Op>
void binary_op(const Tin1 *in1,
               Shape<0> in1_stride,
               const Tin2 *in2,
               Shape<0> in2_stride,
               Tout *out,
               Shape<0> out_stride,
               Shape<0> shape,
               const Op &op) {
  *out = op(*in1, *in2);
}

template <typename Tin1, typename Tin2, typename Tout, int Dim, typename Op>
void binary_op(const Tin1 *in1,
               Shape<Dim> in1_stride,
               const Tin2 *in2,
               Shape<Dim> in2_stride,
               Tout *out,
               Shape<Dim> out_stride,
               Shape<Dim> shape,
               const Op &op) {
  for (Index i = 0; i < shape.head(); ++i) {
    binary_op(
        in1 + i * in1_stride.head(), in1_stride.tail(),
        in2 + i * in2_stride.head(), in2_stride.tail(),
        out + i * out_stride.head(), out_stride.tail(), shape.tail(), op);
  }
}

template <typename Tin1, typename Tin2, typename Tout, int Dim, typename Op>
void binary_op(const ImmutableArray<Tin1, Dim> &in1,
               const ImmutableArray<Tin2, Dim> &in2,
               Array<Tout, Dim> &out,
               const Op &op) {
  binary_op(in1.data(), in1.stride(), in2.data(), in2.stride(),
      out.data(), out.stride(), in1.shape(), op);
}

/** REDUCE ********************************************************************/

template <typename Tin,
          typename Tout,
          typename TransformOp,
          typename ReduceOp>
void reduce(const Tin *in,
            const Shape<0> &stride_in,
            const Shape<0> &shape_in,
            Tout *out,
            const Shape<0> &stride_out,
            const TransformOp &op_t,
            const ReduceOp &op_r) {
  *out = op_r(*out, op_t(*in));
}

template <typename Tin,
          int Dim,
          typename Tout,
          typename TransformOp,
          typename ReduceOp>
void reduce(const Tin *in,
            const Shape<Dim> &stride_in,
            const Shape<Dim> &shape_in,
            Tout *out,
            const Shape<Dim> &stride_out,
            const TransformOp &op_t,
            const ReduceOp &op_r) {
  for (Index i = 0; i < shape_in.head(); ++i) {
    reduce(in + i * stride_in.head(),
           stride_in.tail(),
           shape_in.tail(),
           out + i * stride_out.head(),
           stride_out.tail(),
           op_t,
           op_r);
  }
}

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
  reduce(in.data(), in.stride(), in.shape(), out.data(), stride, op_t, op_r);
}

}  // namespace cpu
}  // namespace impl
}  // namespace aml

