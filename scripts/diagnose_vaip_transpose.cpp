// SPDX-License-Identifier: AGPL-3.0-only
// Exercise the installed VAIP CPU transpose directly, without an NPU session.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  void* lib = dlopen(argv[1], RTLD_LAZY | RTLD_GLOBAL);
  if (!lib) { std::cerr << dlerror() << '\n'; return 2; }
  using Transpose = void (*)(const int16_t*, int16_t*, const std::vector<int64_t>&,
                             const std::vector<int64_t>&);
  auto transpose = reinterpret_cast<Transpose>(dlsym(lib,
      "_ZN18vaip_custom_op_dpu13transpose_i16EPKsPsRKSt6vectorIlSaIlEES7_"));
  if (!transpose) { std::cerr << dlerror() << '\n'; return 2; }
  auto threads = reinterpret_cast<int*>(dlsym(lib,
      "_ZN9morphizen5utils10env_configIi33ENV_PARAM_NUM_OF_TRANSPOSE_THREADE5valueE"));
  std::cout << "transpose_threads=" << (threads ? *threads : -1) << std::endl;
  std::mt19937 random(42);
  int failures = 0;
  int iterations = argc > 2 ? std::atoi(argv[2]) : 10000;
  for (int i = 0; i < iterations; ++i) {
    int hw = i % 3 == 0 ? 56 : i % 3 == 1 ? 28 : 14;
    int channels = i % 3 == 0 ? 64 : i % 3 == 1 ? 128 : 256;
    std::vector<int64_t> shape{1, hw, hw, channels}, perm{0, 3, 1, 2};
    std::vector<int16_t> input(hw*hw*channels), output(input.size()), expected(input.size());
    for (auto& v : input) v = static_cast<int16_t>(random());
    for (int p = 0; p < hw*hw; ++p)
      for (int c = 0; c < channels; ++c)
        expected[c*hw*hw+p] = input[p*channels+c];
    transpose(input.data(), output.data(), shape, perm);
    size_t count = 0, first = 0;
    for (size_t j = 0; j < output.size(); ++j)
      if (output[j] != expected[j]) { if (!count) first = j; ++count; }
    if (count) {
      ++failures;
      if (failures <= 10)
        std::cout << "iteration=" << i << " hw=" << hw << " count=" << count
                  << " first=" << first << " expected=" << expected[first]
                  << " actual=" << output[first] << std::endl;
    }
  }
  std::cout << "iterations=" << iterations << " failures=" << failures << std::endl;
  return failures ? 1 : 0;
}
