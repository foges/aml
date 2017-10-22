#pragma once

#include <aml/array.h>
#include <aml/defs.h>
#include <aml/device.h>
#include <aml/functors.h>
#include <aml/handle.h>
#include <aml/immutable_array.h>
#include <aml/impl/expression.h>

namespace aml {

template <typename OpType, int Dim>
class Expression {
public:
  Expression(const OpType &op, const Shape<Dim> &size, Device device)
    : op_(op), size_(size), device_(device) { }

  OpType op() const {
    return op_;
  }

  Shape<Dim> size() const {
    return size_;
  }

  Device device() const {
    return device_;
  }

private:
  OpType op_;
  Shape<Dim> size_;
  Device device_;
};

// TODO replace by slice
template <typename T, int Dim, typename OpType>
void eval(Handle h, Array<T, Dim> &x, const Expression<OpType, Dim> &expr) {
  AML_ASSERT(x.size() == expr.size(), "Size mismatch");
  AML_ASSERT(x.device() == expr.device(), "Size mismatch");

  AML_DEVICE_EVAL(x.device(), eval(h, x, expr.op()));
}

/** OPS ***********************************************************************/

template <typename OpType1, typename OpType2, typename Op>
class BinaryOp {
public:
  template <int Dim>
  BinaryOp(const Expression<OpType1, Dim> &x,
           const Expression<OpType2, Dim> &y,
           const Op &binary_op)
      : op1_(x.op()), op2_(y.op()), binary_op_(binary_op) { }

  using value_type =
      typename std::result_of<Op(typename OpType1::value_type,
                                 typename OpType2::value_type)>::type;

  template <int Dim>
  value_type operator()(const Shape<Dim> &idx) const {
    return binary_op_(op1_(idx), op2_(idx));
  }

private:
  OpType1 op1_;
  OpType2 op2_;
  Op binary_op_;
};

template <typename OpType1, typename OpType2, typename Op, int Dim>
BinaryOp<OpType1, OpType2, Op>
make_binary_op(const Expression<OpType1, Dim> &x,
               const Expression<OpType2, Dim> &y,
               const Op &binary_op) {
  return { x, y, binary_op };
}

template <typename OpType, typename Op>
class UnaryOp {
public:
  template <int Dim>
  UnaryOp(const Expression<OpType, Dim> &x, const Op &unary_op)
      : op_(x.op()), unary_op_(unary_op) { }

  using value_type =
      typename std::result_of<Op(typename OpType::value_type)>::type;

  template <int Dim>
  value_type operator()(const Shape<Dim> &idx) const {
    return unary_op_(op_(idx));
  }

private:
  OpType op_;
  Op unary_op_;
};

template <typename OpType, typename Op, int Dim>
UnaryOp<OpType, Op>
make_unary_op(const Expression<OpType, Dim> &x,
              const Op &unary_op) {
  return { x, unary_op };
}

template <typename T, int Dim>
class ArrayOp {
public:
  ArrayOp(const ImmutableArray<T, Dim> &x)
    : data_(x.data()), stride_(x.stride()), allocation_(x.allocation()) { }

  using value_type = T;

  T operator()(const Shape<Dim> &idx) const {
    return data_[impl::dot(idx, stride_)];
  }

private:
  const T *data_;
  Shape<Dim> stride_;
  std::shared_ptr<const Allocation> allocation_;
};

template <typename T, int Dim>
Expression<ArrayOp<T, Dim>, Dim>
make_expression(const ImmutableArray<T, Dim> &x) {
  return { ArrayOp<T, Dim>(x), x.size(), x.device() };
}

/** PLUS **********************************************************************/

template <typename OpType1, typename OpType2, int Dim>
Expression<BinaryOp<OpType1, OpType2, Plus>, Dim>
operator+(const Expression<OpType1, Dim> &x,
          const Expression<OpType2, Dim> &y) {
  AML_ASSERT(x.size() == y.size(), "Size mismatch");
  AML_ASSERT(x.device() == y.device(), "Device mismatch");

  return { make_binary_op(x, y, Plus()), x.size(), x.device() };
}

template <typename T, int Dim>
Expression<BinaryOp<ArrayOp<T, Dim>, ArrayOp<T, Dim>, Plus>, Dim>
operator+(const ImmutableArray<T, Dim> &x,
          const ImmutableArray<T, Dim> &y) {
  return make_expression(x) + make_expression(y);
}

template <typename T, int Dim, typename OpType>
Expression<BinaryOp<ArrayOp<T, Dim>, OpType, Plus>, Dim>
operator+(const ImmutableArray<T, Dim> &x,
          const Expression<OpType, Dim> &y) {
  return make_expression(x) + y;
}

template <typename T, int Dim, typename OpType>
Expression<BinaryOp<OpType, ArrayOp<T, Dim>, Plus>, Dim>
operator+(const Expression<OpType, Dim> &x,
          const ImmutableArray<T, Dim> &y) {
  return x + make_expression(y);
}

/** MINUS *********************************************************************/

template <typename OpType1, typename OpType2, int Dim>
Expression<BinaryOp<OpType1, OpType2, Minus>, Dim>
operator-(const Expression<OpType1, Dim> &x,
          const Expression<OpType2, Dim> &y) {
  AML_ASSERT(x.size() == y.size(), "Size mismatch");
  AML_ASSERT(x.device() == y.device(), "Device mismatch");

  return { make_binary_op(x, y, Minus()), x.size(), x.device() };
}

template <typename T, int Dim>
Expression<BinaryOp<ArrayOp<T, Dim>, ArrayOp<T, Dim>, Minus>, Dim>
operator-(const ImmutableArray<T, Dim> &x,
          const ImmutableArray<T, Dim> &y) {
  return make_expression(x) - make_expression(y);
}

template <typename T, int Dim, typename OpType>
Expression<BinaryOp<ArrayOp<T, Dim>, OpType, Minus>, Dim>
operator-(const ImmutableArray<T, Dim> &x,
          const Expression<OpType, Dim> &y) {
  return make_expression(x) - y;
}

template <typename T, int Dim, typename OpType>
Expression<BinaryOp<OpType, ArrayOp<T, Dim>, Minus>, Dim>
operator-(const Expression<OpType, Dim> &x,
          const ImmutableArray<T, Dim> &y) {
  return x - make_expression(y);
}

/** MULTIPLY ******************************************************************/

template <typename OpType1, typename OpType2, int Dim>
Expression<BinaryOp<OpType1, OpType2, Multiply>, Dim>
operator*(const Expression<OpType1, Dim> &x,
          const Expression<OpType2, Dim> &y) {
  AML_ASSERT(x.size() == y.size(), "Size mismatch");
  AML_ASSERT(x.device() == y.device(), "Device mismatch");

  return { make_binary_op(x, y, Multiply()), x.size(), x.device() };
}

template <typename T, int Dim>
Expression<BinaryOp<ArrayOp<T, Dim>, ArrayOp<T, Dim>, Multiply>, Dim>
operator*(const ImmutableArray<T, Dim> &x,
          const ImmutableArray<T, Dim> &y) {
  return make_expression(x) * make_expression(y);
}

template <typename T, int Dim, typename OpType>
Expression<BinaryOp<ArrayOp<T, Dim>, OpType, Multiply>, Dim>
operator*(const ImmutableArray<T, Dim> &x,
          const Expression<OpType, Dim> &y) {
  return make_expression(x) * y;
}

template <typename T, int Dim, typename OpType>
Expression<BinaryOp<OpType, ArrayOp<T, Dim>, Multiply>, Dim>
operator*(const Expression<OpType, Dim> &x,
          const ImmutableArray<T, Dim> &y) {
  return x * make_expression(y);
}

/** DIVIDE ********************************************************************/

template <typename OpType1, typename OpType2, int Dim>
Expression<BinaryOp<OpType1, OpType2, Divide>, Dim>
operator/(const Expression<OpType1, Dim> &x,
          const Expression<OpType2, Dim> &y) {
  AML_ASSERT(x.size() == y.size(), "Size mismatch");
  AML_ASSERT(x.device() == y.device(), "Device mismatch");

  return { make_binary_op(x, y, Divide()), x.size(), x.device() };
}

template <typename T, int Dim>
Expression<BinaryOp<ArrayOp<T, Dim>, ArrayOp<T, Dim>, Divide>, Dim>
operator/(const ImmutableArray<T, Dim> &x,
          const ImmutableArray<T, Dim> &y) {
  return make_expression(x) / make_expression(y);
}

template <typename T, int Dim, typename OpType>
Expression<BinaryOp<ArrayOp<T, Dim>, OpType, Divide>, Dim>
operator/(const ImmutableArray<T, Dim> &x,
          const Expression<OpType, Dim> &y) {
  return make_expression(x) / y;
}

template <typename T, int Dim, typename OpType>
Expression<BinaryOp<OpType, ArrayOp<T, Dim>, Divide>, Dim>
operator/(const Expression<OpType, Dim> &x,
          const ImmutableArray<T, Dim> &y) {
  return x / make_expression(y);
}

}  // namespace aml

