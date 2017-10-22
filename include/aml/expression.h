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

template <typename T>
class ConstOp {
public:
  ConstOp(const T &x) : x_(x) { }

  using value_type = T;

  template <int Dim>
  T operator()(const Shape<Dim>&) const {
    return x_;
  }

private:
  T x_;
};

template <typename T, int Dim>
Expression<ConstOp<T>, Dim>
make_expression(const T &x, const Shape<Dim> &size, Device device) {
  return { ConstOp<T>(x), size, device };
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


template <int Dim,
          typename OpType,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ConstOp<Ts>, OpType, Plus>, Dim>
operator+(const Ts &x,
          const Expression<OpType, Dim> &y) {
  return make_expression(x, y.size(), y.device()) + y;
}

template <int Dim,
          typename OpType,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<OpType, ConstOp<Ts>, Plus>, Dim>
operator+(const Expression<OpType, Dim> &x,
          const Ts &y) {
  return x + make_expression(y, x.size(), x.device());
}

template <typename T,
          int Dim,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ConstOp<Ts>, ArrayOp<T, Dim>, Plus>, Dim>
operator+(const Ts &x,
          const ImmutableArray<T, Dim> &y) {
  return make_expression(x, y.size(), y.device()) + make_expression(y);
}

template <typename T,
          int Dim,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ArrayOp<T, Dim>, ConstOp<Ts>, Plus>, Dim>
operator+(const ImmutableArray<T, Dim> &x,
          const Ts &y) {
  return make_expression(x) + make_expression(y, x.size(), x.device());
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

template <int Dim,
          typename OpType,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ConstOp<Ts>, OpType, Minus>, Dim>
operator-(const Ts &x,
          const Expression<OpType, Dim> &y) {
  return make_expression(x, y.size(), y.device()) - y;
}

template <int Dim,
          typename OpType,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<OpType, ConstOp<Ts>, Minus>, Dim>
operator-(const Expression<OpType, Dim> &x,
          const Ts &y) {
  return x - make_expression(y, x.size(), x.device());
}

template <typename T,
          int Dim,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ConstOp<Ts>, ArrayOp<T, Dim>, Minus>, Dim>
operator-(const Ts &x,
          const ImmutableArray<T, Dim> &y) {
  return make_expression(x, y.size(), y.device()) - make_expression(y);
}

template <typename T,
          int Dim,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ArrayOp<T, Dim>, ConstOp<Ts>, Minus>, Dim>
operator-(const ImmutableArray<T, Dim> &x,
          const Ts &y) {
  return make_expression(x) - make_expression(y, x.size(), x.device());
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

template <int Dim,
          typename OpType,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ConstOp<Ts>, OpType, Multiply>, Dim>
operator*(const Ts &x,
          const Expression<OpType, Dim> &y) {
  return make_expression(x, y.size(), y.device()) *  y;
}

template <int Dim,
          typename OpType,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<OpType, ConstOp<Ts>, Multiply>, Dim>
operator*(const Expression<OpType, Dim> &x,
          const Ts &y) {
  return x * make_expression(y, x.size(), x.device());
}

template <typename T,
          int Dim,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ConstOp<Ts>, ArrayOp<T, Dim>, Multiply>, Dim>
operator*(const Ts &x,
          const ImmutableArray<T, Dim> &y) {
  return make_expression(x, y.size(), y.device()) * make_expression(y);
}

template <typename T,
          int Dim,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ArrayOp<T, Dim>, ConstOp<Ts>, Multiply>, Dim>
operator*(const ImmutableArray<T, Dim> &x,
          const Ts &y) {
  return make_expression(x) * make_expression(y, x.size(), x.device());
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

template <int Dim,
          typename OpType,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ConstOp<Ts>, OpType, Divide>, Dim>
operator/(const Ts &x,
          const Expression<OpType, Dim> &y) {
  return make_expression(x, y.size(), y.device()) /  y;
}

template <int Dim,
          typename OpType,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<OpType, ConstOp<Ts>, Divide>, Dim>
operator/(const Expression<OpType, Dim> &x,
          const Ts &y) {
  return x / make_expression(y, x.size(), x.device());
}

template <typename T,
          int Dim,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ConstOp<Ts>, ArrayOp<T, Dim>, Divide>, Dim>
operator/(const Ts &x,
          const ImmutableArray<T, Dim> &y) {
  return make_expression(x, y.size(), y.device()) / make_expression(y);
}

template <typename T,
          int Dim,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ArrayOp<T, Dim>, ConstOp<Ts>, Divide>, Dim>
operator/(const ImmutableArray<T, Dim> &x,
          const Ts &y) {
  return make_expression(x) / make_expression(y, x.size(), x.device());
}

/** MIN ***********************************************************************/

template <typename OpType1, typename OpType2, int Dim>
Expression<BinaryOp<OpType1, OpType2, Min>, Dim>
min(const Expression<OpType1, Dim> &x,
    const Expression<OpType2, Dim> &y) {
  AML_ASSERT(x.size() == y.size(), "Size mismatch");
  AML_ASSERT(x.device() == y.device(), "Device mismatch");

  return { make_binary_op(x, y, Min()), x.size(), x.device() };
}

template <typename T, int Dim>
Expression<BinaryOp<ArrayOp<T, Dim>, ArrayOp<T, Dim>, Min>, Dim>
min(const ImmutableArray<T, Dim> &x,
    const ImmutableArray<T, Dim> &y) {
  return min(make_expression(x), make_expression(y));
}

template <typename T, int Dim, typename OpType>
Expression<BinaryOp<ArrayOp<T, Dim>, OpType, Min>, Dim>
min(const ImmutableArray<T, Dim> &x,
    const Expression<OpType, Dim> &y) {
  return min(make_expression(x), y);
}

template <typename T, int Dim, typename OpType>
Expression<BinaryOp<OpType, ArrayOp<T, Dim>, Min>, Dim>
min(const Expression<OpType, Dim> &x,
    const ImmutableArray<T, Dim> &y) {
  return min(x, make_expression(y));
}

template <int Dim,
          typename OpType,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ConstOp<Ts>, OpType, Min>, Dim>
min(const Ts &x,
    const Expression<OpType, Dim> &y) {
  return min(make_expression(x, y.size(), y.device()), y);
}

template <int Dim,
          typename OpType,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<OpType, ConstOp<Ts>, Min>, Dim>
min(const Expression<OpType, Dim> &x,
    const Ts &y) {
  return min(x, make_expression(y, x.size(), x.device()));
}

template <typename T,
          int Dim,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ConstOp<Ts>, ArrayOp<T, Dim>, Min>, Dim>
min(const Ts &x,
    const ImmutableArray<T, Dim> &y) {
  return min(make_expression(x, y.size(), y.device()), make_expression(y));
}

template <typename T,
          int Dim,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ArrayOp<T, Dim>, ConstOp<Ts>, Min>, Dim>
min(const ImmutableArray<T, Dim> &x,
    const Ts &y) {
  return min(make_expression(x), make_expression(y, x.size(), x.device()));
}

/** MAX ***********************************************************************/

template <typename OpType1, typename OpType2, int Dim>
Expression<BinaryOp<OpType1, OpType2, Max>, Dim>
max(const Expression<OpType1, Dim> &x,
    const Expression<OpType2, Dim> &y) {
  AML_ASSERT(x.size() == y.size(), "Size mismatch");
  AML_ASSERT(x.device() == y.device(), "Device mismatch");

  return { make_binary_op(x, y, Max()), x.size(), x.device() };
}

template <typename T, int Dim>
Expression<BinaryOp<ArrayOp<T, Dim>, ArrayOp<T, Dim>, Max>, Dim>
max(const ImmutableArray<T, Dim> &x,
    const ImmutableArray<T, Dim> &y) {
  return max(make_expression(x), make_expression(y));
}

template <typename T, int Dim, typename OpType>
Expression<BinaryOp<ArrayOp<T, Dim>, OpType, Max>, Dim>
max(const ImmutableArray<T, Dim> &x,
    const Expression<OpType, Dim> &y) {
  return max(make_expression(x), y);
}

template <typename T, int Dim, typename OpType>
Expression<BinaryOp<OpType, ArrayOp<T, Dim>, Max>, Dim>
max(const Expression<OpType, Dim> &x,
    const ImmutableArray<T, Dim> &y) {
  return max(x, make_expression(y));
}

template <int Dim,
          typename OpType,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ConstOp<Ts>, OpType, Max>, Dim>
max(const Ts &x,
    const Expression<OpType, Dim> &y) {
  return max(make_expression(x, y.size(), y.device()), y);
}

template <int Dim,
          typename OpType,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<OpType, ConstOp<Ts>, Max>, Dim>
max(const Expression<OpType, Dim> &x,
    const Ts &y) {
  return max(x, make_expression(y, x.size(), x.device()));
}

template <typename T,
          int Dim,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ConstOp<Ts>, ArrayOp<T, Dim>, Max>, Dim>
max(const Ts &x,
    const ImmutableArray<T, Dim> &y) {
  return max(make_expression(x, y.size(), y.device()), make_expression(y));
}

template <typename T,
          int Dim,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ArrayOp<T, Dim>, ConstOp<Ts>, Max>, Dim>
max(const ImmutableArray<T, Dim> &x,
    const Ts &y) {
  return max(make_expression(x), make_expression(y, x.size(), x.device()));
}

/** POW ***********************************************************************/

template <typename OpType1, typename OpType2, int Dim>
Expression<BinaryOp<OpType1, OpType2, Pow>, Dim>
pow(const Expression<OpType1, Dim> &x,
    const Expression<OpType2, Dim> &y) {
  AML_ASSERT(x.size() == y.size(), "Size mismatch");
  AML_ASSERT(x.device() == y.device(), "Device mismatch");

  return { make_binary_op(x, y, Pow()), x.size(), x.device() };
}

template <typename T, int Dim>
Expression<BinaryOp<ArrayOp<T, Dim>, ArrayOp<T, Dim>, Pow>, Dim>
pow(const ImmutableArray<T, Dim> &x,
    const ImmutableArray<T, Dim> &y) {
  return pow(make_expression(x), make_expression(y));
}

template <typename T, int Dim, typename OpType>
Expression<BinaryOp<ArrayOp<T, Dim>, OpType, Pow>, Dim>
pow(const ImmutableArray<T, Dim> &x,
    const Expression<OpType, Dim> &y) {
  return pow(make_expression(x), y);
}

template <typename T, int Dim, typename OpType>
Expression<BinaryOp<OpType, ArrayOp<T, Dim>, Pow>, Dim>
pow(const Expression<OpType, Dim> &x,
    const ImmutableArray<T, Dim> &y) {
  return pow(x, make_expression(y));
}

template <int Dim,
          typename OpType,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ConstOp<Ts>, OpType, Pow>, Dim>
pow(const Ts &x,
    const Expression<OpType, Dim> &y) {
  return pow(make_expression(x, y.size(), y.device()), y);
}

template <int Dim,
          typename OpType,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<OpType, ConstOp<Ts>, Pow>, Dim>
pow(const Expression<OpType, Dim> &x,
    const Ts &y) {
  return pow(x, make_expression(y, x.size(), x.device()));
}

template <typename T,
          int Dim,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ConstOp<Ts>, ArrayOp<T, Dim>, Pow>, Dim>
pow(const Ts &x,
    const ImmutableArray<T, Dim> &y) {
  return pow(make_expression(x, y.size(), y.device()), make_expression(y));
}

template <typename T,
          int Dim,
          typename Ts,
          typename = enable_if_t<std::is_arithmetic<Ts>::value>>
Expression<BinaryOp<ArrayOp<T, Dim>, ConstOp<Ts>, Pow>, Dim>
pow(const ImmutableArray<T, Dim> &x,
    const Ts &y) {
  return pow(make_expression(x), make_expression(y, x.size(), x.device()));
}

/** ABS ***********************************************************************/

template <typename OpType, int Dim>
Expression<UnaryOp<OpType, Abs>, Dim>
abs(const Expression<OpType, Dim> &x) {
  return { make_unary_op(x, Abs()), x.size(), x.device() };
}

template <typename T, int Dim>
Expression<UnaryOp<ArrayOp<T, Dim>, Abs>, Dim>
abs(const ImmutableArray<T, Dim> &x) {
  return abs(make_expression(x));
}

/** EXP ***********************************************************************/

template <typename OpType, int Dim>
Expression<UnaryOp<OpType, Exp>, Dim>
exp(const Expression<OpType, Dim> &x) {
  return { make_unary_op(x, Exp()), x.size(), x.device() };
}

template <typename T, int Dim>
Expression<UnaryOp<ArrayOp<T, Dim>, Exp>, Dim>
exp(const ImmutableArray<T, Dim> &x) {
  return exp(make_expression(x));
}

/** LOG ***********************************************************************/

template <typename OpType, int Dim>
Expression<UnaryOp<OpType, Log>, Dim>
log(const Expression<OpType, Dim> &x) {
  return { make_unary_op(x, Log()), x.size(), x.device() };
}

template <typename T, int Dim>
Expression<UnaryOp<ArrayOp<T, Dim>, Log>, Dim>
log(const ImmutableArray<T, Dim> &x) {
  return log(make_expression(x));
}

/** NEGATIVE ******************************************************************/

template <typename OpType, int Dim>
Expression<UnaryOp<OpType, Negative>, Dim>
operator-(const Expression<OpType, Dim> &x) {
  return { make_unary_op(x, Negative()), x.size(), x.device() };
}

template <typename T, int Dim>
Expression<UnaryOp<ArrayOp<T, Dim>, Negative>, Dim>
operator-(const ImmutableArray<T, Dim> &x) {
  return -make_expression(x);
}

/** SQRT **********************************************************************/

template <typename OpType, int Dim>
Expression<UnaryOp<OpType, Sqrt>, Dim>
sqrt(const Expression<OpType, Dim> &x) {
  return { make_unary_op(x, Sqrt()), x.size(), x.device() };
}

template <typename T, int Dim>
Expression<UnaryOp<ArrayOp<T, Dim>, Sqrt>, Dim>
sqrt(const ImmutableArray<T, Dim> &x) {
  return sqrt(make_expression(x));
}

}  // namespace aml

