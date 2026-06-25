#include "marco/Runtime/Printers/CSV/Printer.h"
#include "marco/Runtime/Printers/CSV/CLI.h"
#include "marco/Runtime/Printers/CSV/Options.h"
#include "marco/Runtime/Printers/CSV/Profiler.h"
#include "marco/Runtime/Simulation/Profiler.h"
#include "marco/Runtime/Simulation/Runtime.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <ostream>

using namespace ::marco::runtime;
using namespace ::marco::runtime::printing;

static void configureCSVStream(std::ostream &os) {
  auto &options = printOptions();
  os.precision(options.precision);

  if (options.scientificNotation) {
    os << std::scientific;
  } else {
    os << std::fixed;
  }
}

static void printDerWrapOpening(std::ostream &os, int64_t order) {
  for (int64_t i = 0; i < order; ++i) {
    PRINT_PROFILER_STRING_START
    os << "der(";
    PRINT_PROFILER_STRING_STOP
  }
}

static void printDerWrapClosing(std::ostream &os, int64_t order) {
  for (int64_t i = 0; i < order; ++i) {
    PRINT_PROFILER_STRING_START
    os << ')';
    PRINT_PROFILER_STRING_STOP
  }
}

static void printName(std::ostream &os, char *name, int64_t rank,
                      const int64_t *indices) {
  PRINT_PROFILER_STRING_START
  os << name;
  PRINT_PROFILER_STRING_STOP

  if (rank != 0) {
    assert(indices != nullptr);
    os << '[';

    for (int64_t dim = 0; dim < rank; ++dim) {
      if (dim != 0) {
        PRINT_PROFILER_STRING_START
        os << ',';
        PRINT_PROFILER_STRING_STOP
      }

      // Modelica arrays are 1-based, so increment the printed index by one.
      int64_t index = indices[dim] + 1;

      PRINT_PROFILER_INT_START
      os << index;
      PRINT_PROFILER_INT_STOP
    }

    os << ']';
  }
}

static void printHeader(std::ostream &os, const Simulation &simulation) {
  PRINT_PROFILER_STRING_START
  os << '"' << "time" << '"';
  PRINT_PROFILER_STRING_STOP

  for (int64_t var : simulation.variablesPrintOrder) {
    if (!simulation.printableVariables[var]) {
      // The variable must not be printed.
      continue;
    }

    int64_t rank = simulation.variablesRanks[var];

    if (rank != 0 && simulation.variablesPrintableIndices[var].empty()) {
      // The array variable has no printable indices.
      continue;
    }

    int64_t derOrder = simulation.derOrders[var];
    int64_t baseVar = var;

    for (int64_t i = 0; i < derOrder; ++i) {
      baseVar = simulation.derivativesMap[baseVar];
    }

    assert(baseVar != -1);
    char *name = simulation.variablesNames[baseVar];

    if (rank == 0) {
      // Print only the variable name.
      PRINT_PROFILER_STRING_START
      os << ',' << '"';
      PRINT_PROFILER_STRING_STOP

      printDerWrapOpening(os, derOrder);
      printName(os, name, 0, nullptr);
      printDerWrapClosing(os, derOrder);

      PRINT_PROFILER_STRING_START
      os << '"';
      PRINT_PROFILER_STRING_STOP
    } else {
      // Print the name of the array and the indices, for each possible
      // combination of printable indices.

      for (const auto &range : simulation.variablesPrintableIndices[var]) {
        auto beginIt = MultidimensionalRangeIterator::begin(range);
        auto endIt = MultidimensionalRangeIterator::end(range);

        for (auto it = beginIt; it != endIt; ++it) {
          PRINT_PROFILER_STRING_START
          os << ',' << '"';
          PRINT_PROFILER_STRING_STOP

          printDerWrapOpening(os, derOrder);
          printName(os, name, rank, *it);
          printDerWrapClosing(os, derOrder);

          PRINT_PROFILER_STRING_START
          os << '"';
          PRINT_PROFILER_STRING_STOP
        }
      }
    }
  }

  PRINT_PROFILER_STRING_START
  os << std::endl;
  PRINT_PROFILER_STRING_STOP
}

static void printValueLine(std::ostream &os, const double *values,
                           uint64_t count) {
  configureCSVStream(os);

  for (uint64_t i = 0; i < count; ++i) {
    PRINT_PROFILER_FLOAT_START
    os << values[i];
    PRINT_PROFILER_FLOAT_STOP

    if (i + 1 != count) {
      PRINT_PROFILER_STRING_START
      os << ',';
      PRINT_PROFILER_STRING_STOP
    }
  }

  PRINT_PROFILER_STRING_START
  os << std::endl;
  PRINT_PROFILER_STRING_STOP
}

namespace marco::runtime::printing {
CSVPrinter::CSVPrinter(Simulation *simulation) : Printer(simulation) {}

#ifdef CLI_ENABLE
std::unique_ptr<cli::Category> CSVPrinter::getCLIOptions() {
  return std::make_unique<CommandLineOptions>();
}
#endif // CLI_ENABLE

void CSVPrinter::simulationBegin() {
  openResultFile();

  SIMULATION_PROFILER_PRINTING_START
  ::printHeader(std::cout, *getSimulation());

  if (resultFile.is_open()) {
    ::printHeader(resultFile, *getSimulation());
  }

  SIMULATION_PROFILER_PRINTING_STOP
}

void CSVPrinter::printValues() {
  getBuffer().getActiveBuffer()[0] = getTime();

  for (int64_t var : getSimulation()->variablesPrintOrder) {
    if (!getSimulation()->printableVariables[var]) {
      // The variable must not be printed.
      continue;
    }

    int64_t rank = getSimulation()->variablesRanks[var];

    if (rank != 0 && getSimulation()->variablesPrintableIndices[var].empty()) {
      // The array variable has no printable indices.
      continue;
    }

    if (rank == 0) {
      // Print the scalar variable.
      double value = getVariableValue(var, nullptr);
      getBuffer().getActiveBuffer()[bufferPositions[var]] = value;
    } else {
      // Print the components of the array variable.
      size_t relativePos = 0;

      for (const auto &range :
           getSimulation()->variablesPrintableIndices[var]) {
        auto beginIt = MultidimensionalRangeIterator::begin(range);
        auto endIt = MultidimensionalRangeIterator::end(range);

        for (auto it = beginIt; it != endIt; ++it) {
          double value = getVariableValue(var, *it);
          getBuffer().getActiveBuffer()[bufferPositions[var] + relativePos] =
              value;
          ++relativePos;
        }
      }
    }
  }

  getBuffer().endLine();
}

void CSVPrinter::simulationEnd() {
  getBuffer().flush();

  if (resultFile.is_open()) {
    resultFile.flush();
  }
}

void CSVPrinter::openResultFile() {
  if (resultFile.is_open()) {
    return;
  }

  std::string path = printOptions().resultFile;

  if (path.empty()) {
    // 中文：默认写入 OMC 风格的结果文件，同时保持现有 stdout 输出不变。
    // English: By default, write an OMC-style result file while preserving the
    // existing stdout stream unchanged.
    path = std::string(getModelName()) + "_res.csv";
  }

  resultFile.open(path, std::ios::out | std::ios::trunc);

  if (!resultFile.is_open()) {
    std::cerr << "warning: unable to open CSV result file '" << path << "'"
              << std::endl;
  }
}

void CSVPrinter::initialize() {
  buffer = DoubleBuffer(1 + getSimulation()->getNumOfPrintableScalarVariables(),
                        printOptions().bufferSize,
                        [&](const double *values, uint64_t count) {
                          printBufferedValues(values, count);
                        });

  bufferPositions.resize(getSimulation()->printableVariables.size(), 0);

  // Time variable.
  bufferPositions.push_back(0);

  // Model variables.
  int64_t position = 1;

  for (int64_t variable : getSimulation()->variablesPrintOrder) {
    bufferPositions[variable] = position;
    position += getSimulation()->getNumOfPrintableScalarVariables(variable);
  }
}

DoubleBuffer &CSVPrinter::getBuffer() {
  if (!buffer) {
    initialize();
  }

  return *buffer;
}

void CSVPrinter::printBufferedValues(const double *values, uint64_t count) {
  SIMULATION_PROFILER_PRINTING_START

  ::printValueLine(std::cout, values, count);

  if (resultFile.is_open()) {
    // 中文：文件和 stdout 共用同一批缓冲值，保证两边列顺序和数值格式一致。
    // English: Reuse the same buffered values for the file and stdout so column
    // order and numeric formatting stay identical.
    ::printValueLine(resultFile, values, count);
  }

  SIMULATION_PROFILER_PRINTING_STOP
}
} // namespace marco::runtime::printing

namespace marco::runtime {
std::unique_ptr<Printer> getPrinter(Simulation *simulation) {
  return std::make_unique<printing::CSVPrinter>(simulation);
}
} // namespace marco::runtime
