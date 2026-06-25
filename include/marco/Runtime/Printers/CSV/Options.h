#ifndef MARCO_RUNTIME_PRINTING_CONFIG_H
#define MARCO_RUNTIME_PRINTING_CONFIG_H

#include <cstddef>
#include <string>

namespace marco::runtime::printing {
struct PrintOptions {
  bool scientificNotation = false;
  unsigned int precision = 9;

  // Buffer size in bytes.
  size_t bufferSize = 10 * 1024 * 1024;

  // 中文：空路径表示使用 "<model>_res.csv"，保持和 OpenModelica 默认命名一致。
  // English: An empty path means "<model>_res.csv", matching OpenModelica's
  // default result-file naming.
  std::string resultFile;
};

PrintOptions &printOptions();
} // namespace marco::runtime::printing

#endif // MARCO_RUNTIME_PRINTING_CONFIG_H
