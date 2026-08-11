add_subdirectory(EulerForward)
add_subdirectory(RungeKutta)

add_subdirectory(SUNDIALS)

# 中文：IDA 的 component-local closure 复用 KINSOL 稀疏内核，故先构建 KINSOL。
# English: IDA component-local closure reuses the sparse KINSOL core, so build KINSOL first.
add_subdirectory(KINSOL)
add_subdirectory(IDA)
