#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include <aml/defs.h>

namespace aml {
namespace impl {

using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

struct Profile {
  Profile() : duration_us(0), num_calls(0) { }

  uint64_t duration_us;
  uint32_t num_calls;
};

class Toc {
public:
  Toc() : enabled_(false) { }

  Toc(const time_point &start,
      Profile* profile,
      std::function<void()> sync)
      : enabled_(true),
        did_stop_(false),
        start_(start),
        profile_(profile),
        sync_(sync) {
      sync_();
  }

  ~Toc() {
    AML_ASSERT(!enabled_ || did_stop_, "Must stop profiler");
  }

  void stop() {
    if (!enabled_) {
      return;
    }

    AML_ASSERT(!did_stop_, "Multiple invocations of stop for same profile");
    did_stop_ = true;
    sync_();
    auto elapsed = std::chrono::high_resolution_clock::now() - start_;
    profile_->duration_us +=
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    profile_->num_calls += 1;
  }

private:
  bool enabled_;
  bool did_stop_;
  time_point start_;
  Profile *profile_;
  std::function<void()> sync_;
};

class Profiler {
public:
  Profiler(bool enabled) : enabled_(enabled) { }

  Toc tic(const std::string &name) {
    return tic(name, []{});
  }

  Toc tic(const std::string &name, const std::function<void()> &sync) {
    if (!enabled_) {
      return Toc();
    }

    auto it = profiles_.find(name);
    if (it == profiles_.end()) {
      it = profiles_.insert(
          std::make_pair(name, std::unique_ptr<Profile>(new Profile()))).first;
    }
    auto now = std::chrono::high_resolution_clock::now();
    return Toc(now, it->second.get(), sync);
  }

  std::string to_string() const {
    using element = std::tuple<uint64_t, uint32_t, std::string>;
    std::vector<element> elements;
    elements.reserve(profiles_.size());

    for (auto &val : profiles_) {
      elements.push_back(std::make_tuple(
          val.second->duration_us, val.second->num_calls, val.first));
    }

    std::sort(elements.begin(), elements.end(),
        [](const element &x, element &y){ return x > y; });

    std::stringstream ss;
    ss << "-------------------------------------------------------------------"
       << std::endl;
    ss << "| "
       << std::setw(19) << "Name"      << " | "
       << std::setw(19) << "Time (ms)" << " | "
       << std::setw(19) << "Num calls" << " |"
       << std::endl;
    ss << "|-----------------------------------------------------------------|"
       << std::endl;

    for (auto &val : elements) {
      ss << "| "
         << std::setw(19) << std::get<2>(val) << " | "
         << std::setw(19) << std::get<0>(val) / 1000 << " | "
         << std::setw(19) << std::get<1>(val) << " |"
         << std::endl;
    }
    ss << "-------------------------------------------------------------------"
       << std::endl;

    return ss.str();
  }

private:
  bool enabled_;
  std::map<std::string, std::unique_ptr<Profile>> profiles_;
};

}  // namespace impl
}  // namespace aml

