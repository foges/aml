#pragma once

#include <math.h>

#include <aml/defs.h>

namespace aml {

class Abs {
public:
  template <typename T>
  AML_HOST_DEVICE T operator()(const T &x) const {
    return x < 0 ? -x : x;
  }
};

class Divide {
public:
  template <typename T>
  AML_HOST_DEVICE T operator()(const T &x, const T &y) const {
    return x / y;
  }
};


class Exp {
public:
  AML_HOST_DEVICE float operator()(const float &x) const {
    return expf(x);
  }

  AML_HOST_DEVICE double operator()(const double &x) const {
    return exp(x);
  }
};

class Identity {
public:
  template <typename T>
  AML_HOST_DEVICE T operator()(const T &x) const {
    return x;
  }
};

class Inv {
public:
  template <typename T>
  AML_HOST_DEVICE T operator()(const T &x) const {
    return static_cast<T>(1) / x;
  }
};

class Log {
public:
  AML_HOST_DEVICE float operator()(const float &x) const {
    return logf(x);
  }

  AML_HOST_DEVICE double operator()(const double &x) const {
    return log(x);
  }
};

class Max {
public:
  template <typename T>
  AML_HOST_DEVICE T operator()(const T &x, const T &y) const {
    return x > y ? x : y;
  }
};

class Min {
public:
  template <typename T>
  AML_HOST_DEVICE T operator()(const T &x, const T &y) const {
    return x < y ? x : y;
  }
};

class Minus {
public:
  template <typename T>
  AML_HOST_DEVICE T operator()(const T &x, const T &y) const {
    return x - y;
  }
};

class Multiply {
public:
  template <typename T>
  AML_HOST_DEVICE T operator()(const T &x, const T &y) const {
    return x * y;
  }
};

class Plus {
public:
  template <typename T>
  AML_HOST_DEVICE T operator()(const T &x, const T &y) const {
    return x + y;
  }
};

class Pow {
public:
  AML_HOST_DEVICE float operator()(const float &x, const float &y) const {
    return powf(x, y);
  }

  AML_HOST_DEVICE double operator()(const double &x, const double &y) const {
    return pow(x, y);
  }
};

class Sqrt {
public:
  AML_HOST_DEVICE float operator()(const float &x) const {
    return sqrtf(x);
  }

  AML_HOST_DEVICE double operator()(const double &x) const {
    return sqrt(x);
  }
};

class Square {
public:
  template <typename T>
  AML_HOST_DEVICE T operator()(const T &x) const {
    return x * x;
  }
};

}  // namespace aml

