#ifndef MARCO_RUNTIME_IDA_C2KERNELV1_H
#define MARCO_RUNTIME_IDA_C2KERNELV1_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 中文：可选编译器/Runtime 协议独立于私有 C++ 布局，IDA 初始化前复制并验证不可变描述。
//       各数组按声明行数提供，J 为行数平方；回调身份包含缺失 J 条目，绑定真实标量注册，不能只凭名字、hash、端点大小或数值零。
// Optional compiler/Runtime protocol, independent of private C++ layouts.
// Registration copies and validates this immutable description before IDA
// initialization. All arrays have the declared row count (J has rows*rows).
// Callback identities include absent J entries; they bind the exact existing
// scalar registration, not merely a name, hash, endpoint size or numeric zero.
typedef void (*MarcoC2ResidualV1)(double, uint64_t, double *);
typedef void (*MarcoC2JacobianV1)(double, uint64_t, uint64_t, double *);
typedef void (*MarcoC2InstallV1)(const double *);
typedef void (*MarcoC2ReadV1)(double *);
typedef int32_t (*MarcoC2HealthV1)(const double *, const double *,
                                 const double *, double *);
typedef int32_t (*MarcoC2SolveV1)(
    double, double *, const double *, const double *, double, uint64_t,
    const uint64_t *, const uint64_t *, const uint64_t *, const uint64_t *,
    double *, uint64_t *);

// 中文：V1 显式声明大小、回调和组件/epoch 布局，不依赖私有 C++ 对象布局；旧标量接口仍有效。
// English: V1 declares descriptor size, callbacks and component/epoch layout explicitly,
// independently of private C++ objects; legacy scalar interfaces remain valid.
typedef struct MarcoC2KernelV1 {
  uint32_t version;
  uint32_t descriptorBytes;
  uint64_t domain;
  uint64_t epoch;
  uint64_t component;
  uint64_t rows;
  uint64_t scratchBytes;
  const void *const *residuals;
  const void *const *jacobians;
  const void *const *getters;
  const void *const *setters;
  const uint64_t *matchedColumns;
  MarcoC2ResidualV1 residual;
  MarcoC2JacobianV1 jacobian;
  MarcoC2InstallV1 install;
  MarcoC2ReadV1 read;
  MarcoC2HealthV1 health;
  MarcoC2SolveV1 solve;
} MarcoC2KernelV1;

// owner is the existing opaque IDA handle. Version/layout/binding errors are
// rejected before solving. Numerical misses use the unchanged original
// solver from its original warm start; no value survives a trial or epoch.
// 中文：初始化前校验并复制描述符；当前数值点不安全时从原 warm start 进入原求解路径。
// English: Validate and copy the descriptor before initialization; an unsafe numerical point
// returns to the original solver with the original warm start.
void _MidaRegisterC2KernelV1(void *owner, const MarcoC2KernelV1 *descriptor);

#ifdef __cplusplus
}
#endif
#endif
