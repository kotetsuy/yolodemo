// SPDX-License-Identifier: AGPL-3.0-only
// Diagnostic-only interposer: record output BO mappings and optionally flush
// their CPU cache lines after session.run, before the NPU can reuse them.
#include <xrt/xrt_bo.h>
#include <dlfcn.h>
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

struct Region { char* address; size_t bytes; size_t offset; xrt::bo buffer; };
static std::vector<Region> regions;
static std::mutex lock;
static size_t pdi_calls = 0;
static size_t post_sync_calls = 0;

using Sync = void (*)(xrt::bo*, xclBOSyncDirection, size_t, size_t);
extern "C" void intercepted_sync(xrt::bo*, xclBOSyncDirection, size_t, size_t)
    asm("_ZN3xrt2bo4syncE18xclBOSyncDirectionmm");
extern "C" void intercepted_sync(xrt::bo* bo, xclBOSyncDirection direction,
                                 size_t bytes, size_t offset) {
  static auto original = reinterpret_cast<Sync>(dlsym(RTLD_NEXT,
      "_ZN3xrt2bo4syncE18xclBOSyncDirectionmm"));
  original(bo, direction, bytes, offset);
  if (direction == XCL_BO_SYNC_BO_FROM_DEVICE) {
    auto* address = bo->map<char*>() + offset;
    std::lock_guard<std::mutex> guard(lock);
    for (auto& r : regions) if (r.address == address && r.bytes == bytes) return;
    regions.push_back({address, bytes, offset, *bo});
  }
}

extern "C" size_t diagnostic_flush_outputs(int flush) {
  std::lock_guard<std::mutex> guard(lock);
  size_t total = 0;
  for (const auto& r : regions) {
    total += r.bytes;
    if (flush) {
      uintptr_t end = reinterpret_cast<uintptr_t>(r.address) + r.bytes;
      for (uintptr_t p = reinterpret_cast<uintptr_t>(r.address) & ~uintptr_t(63);
           p < end; p += 64) _mm_clflush(reinterpret_cast<void*>(p));
    }
  }
  if (flush) _mm_mfence();
  return total;
}

// The installed runner syncs the output BO before run_pdi. Test an additional
// invalidation after device completion, before CPU depadding reads the BO.
extern "C" void intercepted_pdi(void*, int)
    asm("_ZN15DpuKernelRunner7run_pdiEi");
extern "C" void intercepted_pdi(void* runner, int index) {
  using Run = void (*)(void*, int);
  static auto original = reinterpret_cast<Run>(dlsym(RTLD_NEXT,
      "_ZN15DpuKernelRunner7run_pdiEi"));
  original(runner, index);
  ++pdi_calls;
  if (auto* mode = std::getenv("DEMO_DIAG_SYNC_AFTER_NPU")) {
    if (std::strcmp(mode, "driver") == 0) {
      static auto sync = reinterpret_cast<Sync>(dlsym(RTLD_NEXT,
          "_ZN3xrt2bo4syncE18xclBOSyncDirectionmm"));
      std::lock_guard<std::mutex> guard(lock);
      for (auto& r : regions)
        sync(&r.buffer, XCL_BO_SYNC_BO_FROM_DEVICE, r.bytes, r.offset);
    } else {
      diagnostic_flush_outputs(1);
    }
    ++post_sync_calls;
  }
}

extern "C" size_t diagnostic_pdi_calls() { return pdi_calls; }
extern "C" size_t diagnostic_post_sync_calls() { return post_sync_calls; }
