#ifndef MARCO_RUNTIME_PRINTING_PRINTERS_CSV_H
#define MARCO_RUNTIME_PRINTING_PRINTERS_CSV_H

#include "marco/Runtime/Printers/DoubleBuffer.h"
#include "marco/Runtime/Printers/Printer.h"
#include <fstream>
#include <optional>

namespace marco::runtime::printing {
class CSVPrinter : public Printer {
public:
  CSVPrinter(Simulation *simulation);

#ifdef CLI_ENABLE
  std::unique_ptr<cli::Category> getCLIOptions() override;
#endif // CLI_ENABLE

  void simulationBegin() override;

  void printValues() override;

  void simulationEnd() override;

private:
  void initialize();

  void openResultFile();

  DoubleBuffer &getBuffer();

  void printBufferedValues(const double *values, uint64_t count);

private:
  // 中文：结果输出按需创建缓冲；关闭输出不分配它，但仿真、代数量更新和失败传播照常执行。
  // English: Result output creates its buffer on demand; disabling output skips allocation
  // while preserving simulation, algebraic updates and failure propagation.
  std::optional<DoubleBuffer> buffer{std::nullopt};
  std::vector<size_t> bufferPositions;
  std::ofstream resultFile;
};
} // namespace marco::runtime::printing

#endif // MARCO_RUNTIME_PRINTING_PRINTERS_CSV_H
