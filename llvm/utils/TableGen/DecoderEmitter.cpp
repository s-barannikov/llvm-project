//===---------------- DecoderEmitter.cpp - Decoder Generator --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// It contains the tablegen backend that emits the decoder functions for
// targets with fixed/variable length instruction set.
//
//===----------------------------------------------------------------------===//

#include "Common/CodeGenHwModes.h"
#include "Common/CodeGenInstruction.h"
#include "Common/CodeGenTarget.h"
#include "Common/InfoByHwMode.h"
#include "Common/InstructionEncoding.h"
#include "Common/VarLenCodeEmitterGen.h"
#include "DecoderTree.h"
#include "TableGenBackends.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "decoder-emitter"

extern cl::OptionCategory DisassemblerEmitterCat;

enum SuppressLevel {
  SUPPRESSION_DISABLE,
  SUPPRESSION_LEVEL1,
  SUPPRESSION_LEVEL2
};

static cl::opt<SuppressLevel> DecoderEmitterSuppressDuplicates(
    "suppress-per-hwmode-duplicates",
    cl::desc("Suppress duplication of instrs into per-HwMode decoder tables"),
    cl::values(
        clEnumValN(
            SUPPRESSION_DISABLE, "O0",
            "Do not prevent DecoderTable duplications caused by HwModes"),
        clEnumValN(
            SUPPRESSION_LEVEL1, "O1",
            "Remove duplicate DecoderTable entries generated due to HwModes"),
        clEnumValN(
            SUPPRESSION_LEVEL2, "O2",
            "Extract HwModes-specific instructions into new DecoderTables, "
            "significantly reducing Table Duplications")),
    cl::init(SUPPRESSION_DISABLE), cl::cat(DisassemblerEmitterCat));

namespace llvm {
cl::opt<bool> UseFnTableInDecodeToMCInst(
    "use-fn-table-in-decode-to-mcinst",
    cl::desc(
        "Use a table of function pointers instead of a switch case in the\n"
        "generated `decodeToMCInst` function. Helps improve compile time\n"
        "of the generated code."),
    cl::init(false), cl::cat(DisassemblerEmitterCat));

// Enabling this option requires use of different `InsnType` for different
// bitwidths and defining `InsnBitWidth` template specialization for the
// `InsnType` types used. Some common specializations are already defined in
// MCDecoder.h.
cl::opt<bool> SpecializeDecodersPerBitwidth(
    "specialize-decoders-per-bitwidth",
    cl::desc("Specialize the generated `decodeToMCInst` function per bitwidth. "
             "Helps reduce the code size."),
    cl::init(false), cl::cat(DisassemblerEmitterCat));

cl::opt<bool> IgnoreNonDecodableOperands(
    "ignore-non-decodable-operands",
    cl::desc(
        "Do not issue an error if an operand cannot be decoded automatically."),
    cl::init(false), cl::cat(DisassemblerEmitterCat));

cl::opt<bool> IgnoreFullyDefinedOperands(
    "ignore-fully-defined-operands",
    cl::desc(
        "Do not automatically decode operands with no '?' in their encoding."),
    cl::init(false), cl::cat(DisassemblerEmitterCat));
} // namespace llvm

STATISTIC(NumEncodings, "Number of encodings considered");
STATISTIC(NumEncodingsLackingDisasm,
          "Number of encodings without disassembler info");
STATISTIC(NumInstructions, "Number of instructions considered");
STATISTIC(NumEncodingsSupported, "Number of encodings supported");
STATISTIC(NumEncodingsOmitted, "Number of encodings omitted");

namespace {

using NamespacesHwModesMap = std::map<StringRef, std::set<unsigned>>;

class DecoderEmitter {
  const RecordKeeper &RK;
  CodeGenTarget Target;
  const CodeGenHwModes &CGH;

  /// All parsed encodings.
  std::vector<InstructionEncoding> Encodings;

  /// Encodings IDs for each HwMode. An ID is an index into Encodings.
  SmallDenseMap<unsigned, std::vector<unsigned>> EncodingIDsByHwMode;

public:
  explicit DecoderEmitter(const RecordKeeper &RK);

  const CodeGenTarget &getTarget() const { return Target; }

  void emitInstrLenTable(formatted_raw_ostream &OS,
                         ArrayRef<unsigned> InstrLen) const;
  void emitPredicateFunction(formatted_raw_ostream &OS,
                             const PredicateSet &Predicates) const;
  void emitDecoderFunction(formatted_raw_ostream &OS,
                           const DecoderSet &Decoders,
                           unsigned BucketBitWidth) const;

  // run - Output the code emitter
  void run(raw_ostream &o) const;

private:
  void collectHwModesReferencedForEncodings(
      std::vector<unsigned> &HwModeIDs,
      NamespacesHwModesMap &NamespacesWithHwModes) const;

  void
  handleHwModesUnrelatedEncodings(unsigned EncodingID,
                                  ArrayRef<unsigned> HwModeIDs,
                                  NamespacesHwModesMap &NamespacesWithHwModes);

  void parseInstructionEncodings();
};

} // namespace

void DecoderEmitter::emitInstrLenTable(formatted_raw_ostream &OS,
                                       ArrayRef<unsigned> InstrLen) const {
  OS << "static const uint8_t InstrLenTable[] = {\n";
  for (unsigned Len : InstrLen)
    OS << Len << ",\n";
  OS << "};\n\n";
}

void DecoderEmitter::emitPredicateFunction(
    formatted_raw_ostream &OS, const PredicateSet &Predicates) const {
  // The predicate function is just a big switch statement based on the
  // input predicate index.
  OS << "static bool checkDecoderPredicate(unsigned Idx, const FeatureBitset "
        "&FB) {\n";
  OS << "  switch (Idx) {\n";
  OS << "  default: llvm_unreachable(\"Invalid index!\");\n";
  for (const auto &[Index, Predicate] : enumerate(Predicates)) {
    OS << "  case " << Index << ":\n";
    OS << "    return " << Predicate << ";\n";
  }
  OS << "  }\n";
  OS << "}\n\n";
}

void DecoderEmitter::emitDecoderFunction(formatted_raw_ostream &OS,
                                         const DecoderSet &Decoders,
                                         unsigned BucketBitWidth) const {
  // The decoder function is just a big switch statement or a table of function
  // pointers based on the input decoder index.

  // TODO: When InsnType is large, using uint64_t limits all fields to 64 bits
  // It would be better for emitBinaryParser to use a 64-bit tmp whenever
  // possible but fall back to an InsnType-sized tmp for truly large fields.
  StringRef TmpTypeDecl =
      "using TmpType = std::conditional_t<std::is_integral<InsnType>::value, "
      "InsnType, uint64_t>;\n";
  StringRef DecodeParams =
      "DecodeStatus S, InsnType insn, MCInst &MI, uint64_t Address, const "
      "MCDisassembler *Decoder, bool &DecodeComplete";

  // Print the name of the decode function to OS.
  auto PrintDecodeFnName = [&OS, BucketBitWidth](unsigned DecodeIdx) {
    OS << "decodeFn";
    if (BucketBitWidth != 0) {
      OS << '_' << BucketBitWidth << "bit";
    }
    OS << '_' << DecodeIdx;
  };

  // Print the template statement.
  auto PrintTemplate = [&OS, BucketBitWidth]() {
    OS << "template <typename InsnType>\n";
    OS << "static ";
    if (BucketBitWidth != 0)
      OS << "std::enable_if_t<InsnBitWidth<InsnType> == " << BucketBitWidth
         << ", DecodeStatus>\n";
    else
      OS << "DecodeStatus ";
  };

  if (UseFnTableInDecodeToMCInst) {
    // Emit a function for each case first.
    for (const auto &[Index, Decoder] : enumerate(Decoders)) {
      PrintTemplate();
      PrintDecodeFnName(Index);
      OS << "(" << DecodeParams << ") {\n";
      OS << "  " << TmpTypeDecl;
      OS << "  [[maybe_unused]] TmpType tmp;\n";
      OS << Decoder;
      OS << "  return S;\n";
      OS << "}\n\n";
    }
  }

  OS << "// Handling " << Decoders.size() << " cases.\n";
  PrintTemplate();
  OS << "decodeToMCInst(unsigned Idx, " << DecodeParams << ") {\n";
  OS << "  DecodeComplete = true;\n";

  if (UseFnTableInDecodeToMCInst) {
    // Build a table of function pointers
    OS << "  using DecodeFnTy = DecodeStatus (*)(" << DecodeParams << ");\n";
    OS << "  static constexpr DecodeFnTy decodeFnTable[] = {\n";
    for (size_t Index : llvm::seq(Decoders.size())) {
      OS << "    ";
      PrintDecodeFnName(Index);
      OS << ",\n";
    }
    OS << "  };\n";
    OS << "  if (Idx >= " << Decoders.size() << ")\n";
    OS << "    llvm_unreachable(\"Invalid decoder index!\");\n";
    OS << "  return decodeFnTable[Idx](S, insn, MI, Address, Decoder, "
          "DecodeComplete);\n";
  } else {
    OS << "  " << TmpTypeDecl;
    OS << "  TmpType tmp;\n";
    OS << "  switch (Idx) {\n";
    OS << "  default: llvm_unreachable(\"Invalid decoder index!\");\n";
    for (const auto &[Index, Decoder] : enumerate(Decoders)) {
      OS << "  case " << Index << ":\n";
      OS << Decoder;
      OS << "    return S;\n";
    }
    OS << "  }\n";
  }
  OS << "}\n";
}

// emitDecodeInstruction - Emit the templated helper function
// decodeInstruction().
static void emitDecodeInstruction(formatted_raw_ostream &OS, bool IsVarLenInst,
                                  const DecoderTableInfo &TableInfo) {
  OS << R"(
template <typename InsnType>
static DecodeStatus decodeInstruction(const uint8_t DecodeTable[], MCInst &MI,
                                      InsnType insn, uint64_t Address,
                                      const MCDisassembler *DisAsm,
                                      const MCSubtargetInfo &STI)";
  if (IsVarLenInst) {
    OS << ",\n                                      "
          "llvm::function_ref<void(APInt &, uint64_t)> makeUp";
  }
  OS << ") {\n";
  if (TableInfo.HasCheckPredicate)
    OS << "  const FeatureBitset &Bits = STI.getFeatureBits();\n";
  OS << "  const uint8_t *Ptr = DecodeTable;\n";

  if (SpecializeDecodersPerBitwidth) {
    // Fail with a fatal error if decoder table's bitwidth does not match
    // `InsnType` bitwidth.
    OS << R"(
  [[maybe_unused]] uint32_t BitWidth = decodeULEB128AndIncUnsafe(Ptr);
  assert(InsnBitWidth<InsnType> == BitWidth &&
         "Table and instruction bitwidth mismatch");
)";
  }

  OS << R"(
  SmallVector<const uint8_t *, 8> ScopeStack;
  DecodeStatus S = MCDisassembler::Success;
  while (true) {
    ptrdiff_t Loc = Ptr - DecodeTable;
    const uint8_t DecoderOp = *Ptr++;
    switch (DecoderOp) {
    default:
      errs() << Loc << ": Unexpected decode table opcode: "
             << (int)DecoderOp << '\n';
      return MCDisassembler::Fail;
    case OPC_Scope: {
      unsigned NumToSkip = decodeULEB128AndIncUnsafe(Ptr);
      const uint8_t *SkipTo = Ptr + NumToSkip;
      ScopeStack.push_back(SkipTo);
      LLVM_DEBUG(dbgs() << Loc << ": OPC_Scope(" << SkipTo - DecodeTable
                        << ")\n");
      break;
    }
    case OPC_SwitchField: {
      // Decode the start value.
      unsigned Start = decodeULEB128AndIncUnsafe(Ptr);
      unsigned Len = *Ptr++;)";
  if (IsVarLenInst)
    OS << "\n      makeUp(insn, Start + Len);";
  OS << R"(
      uint64_t FieldValue = fieldFromInstruction(insn, Start, Len);
      uint64_t CaseValue;
      unsigned CaseSize;
      while (true) {
        CaseValue = decodeULEB128AndIncUnsafe(Ptr);
        CaseSize = decodeULEB128AndIncUnsafe(Ptr);
        if (FieldValue == CaseValue || !CaseSize)
          break;
        Ptr += CaseSize;
      }
      if (FieldValue == CaseValue) {
        LLVM_DEBUG(dbgs() << Loc << ": OPC_SwitchField(" << Start << ", " << Len
                          << "): " << FieldValue << '\n');
      } else {
        if (ScopeStack.empty()) {
          LLVM_DEBUG(dbgs() << "returning Fail\n");
          return MCDisassembler::Fail;
        }
        Ptr = ScopeStack.pop_back_val();
        LLVM_DEBUG(dbgs() << "continuing at " << Ptr - DecodeTable << '\n');
      }
      break;
    }
    case OPC_CheckField: {
      // Decode the start value.
      unsigned Start = decodeULEB128AndIncUnsafe(Ptr);
      unsigned Len = *Ptr;)";
  if (IsVarLenInst)
    OS << "\n      makeUp(insn, Start + Len);";
  OS << R"(
      uint64_t FieldValue = fieldFromInstruction(insn, Start, Len);
      // Decode the field value.
      unsigned PtrLen = 0;
      uint64_t ExpectedValue = decodeULEB128(++Ptr, &PtrLen);
      Ptr += PtrLen;
      bool Failed = ExpectedValue != FieldValue;

      LLVM_DEBUG(dbgs() << Loc << ": OPC_CheckField(" << Start << ", " << Len
                        << ", " << ExpectedValue << "): FieldValue = "
                        << FieldValue << ", ExpectedValue = " << ExpectedValue
                        << ": " << (Failed ? "FAIL, " : "PASS\n"););
      if (Failed) {
        if (ScopeStack.empty()) {
          LLVM_DEBUG(dbgs() << "returning Fail\n");
          return MCDisassembler::Fail;
        }
        Ptr = ScopeStack.pop_back_val();
        LLVM_DEBUG(dbgs() << "continuing at " << Ptr - DecodeTable << '\n');
      }
      break;
    })";
  if (TableInfo.HasCheckPredicate) {
    OS << R"(
    case OPC_CheckPredicate: {
      // Decode the Predicate Index value.
      unsigned PIdx = decodeULEB128AndIncUnsafe(Ptr);
      // Check the predicate.
      bool Failed = !checkDecoderPredicate(PIdx, Bits);

      LLVM_DEBUG(dbgs() << Loc << ": OPC_CheckPredicate(" << PIdx << "): "
                        << (Failed ? "FAIL, " : "PASS\n"););

      if (Failed) {
        if (ScopeStack.empty()) {
          LLVM_DEBUG(dbgs() << "returning Fail\n");
          return MCDisassembler::Fail;
        }
        Ptr = ScopeStack.pop_back_val();
        LLVM_DEBUG(dbgs() << "continuing at " << Ptr - DecodeTable << '\n');
      }
      break;
    })";
  }
  OS << R"(
    case OPC_Decode: {
      // Decode the Opcode value.
      unsigned Opc = decodeULEB128AndIncUnsafe(Ptr);
      unsigned DecodeIdx = decodeULEB128AndIncUnsafe(Ptr);

      MI.clear();
      MI.setOpcode(Opc);
      bool DecodeComplete;)";
  if (IsVarLenInst) {
    OS << "\n      unsigned Len = InstrLenTable[Opc];\n"
       << "      makeUp(insn, Len);";
  }
  OS << R"(
      S = decodeToMCInst(DecodeIdx, S, insn, MI, Address, DisAsm, DecodeComplete);
      assert(DecodeComplete);

      LLVM_DEBUG(dbgs() << Loc << ": OPC_Decode: opcode " << Opc
                   << ", using decoder " << DecodeIdx << ": "
                   << (S != MCDisassembler::Fail ? "PASS\n" : "FAIL\n"));
      return S;
    })";
  if (TableInfo.HasTryDecode) {
    OS << R"(
    case OPC_TryDecode: {
      // Decode the Opcode value.
      unsigned Opc = decodeULEB128AndIncUnsafe(Ptr);
      unsigned DecodeIdx = decodeULEB128AndIncUnsafe(Ptr);

      // Perform the decode operation.
      MCInst TmpMI;
      TmpMI.setOpcode(Opc);
      bool DecodeComplete;
      S = decodeToMCInst(DecodeIdx, S, insn, TmpMI, Address, DisAsm, DecodeComplete);
      LLVM_DEBUG(dbgs() << Loc << ": OPC_TryDecode: opcode " << Opc
                   << ", using decoder " << DecodeIdx << ": ");

      if (DecodeComplete) {
        // Decoding complete.
        LLVM_DEBUG(dbgs() << (S != MCDisassembler::Fail ? "PASS\n" : "FAIL\n"));
        MI = TmpMI;
        return S;
      }
      assert(S == MCDisassembler::Fail);
      if (ScopeStack.empty()) {
        LLVM_DEBUG(dbgs() << "FAIL, returning FAIL\n");
        return MCDisassembler::Fail;
      }
      Ptr = ScopeStack.pop_back_val();
      LLVM_DEBUG(dbgs() << "FAIL, continuing at " << Ptr - DecodeTable << '\n');
      // Reset decode status. This also drops a SoftFail status that could be
      // set before the decode attempt.
      S = MCDisassembler::Success;
      break;
    })";
  }
  if (TableInfo.HasSoftFail) {
    OS << R"(
    case OPC_SoftFail: {
      // Decode the mask values.
      uint64_t PositiveMask = decodeULEB128AndIncUnsafe(Ptr);
      uint64_t NegativeMask = decodeULEB128AndIncUnsafe(Ptr);
      bool Failed = (insn & PositiveMask) != 0 || (~insn & NegativeMask) != 0;
      if (Failed)
        S = MCDisassembler::SoftFail;
      LLVM_DEBUG(dbgs() << Loc << ": OPC_SoftFail: " << (Failed ? "FAIL\n" : "PASS\n"));
      break;
    })";
  }
  OS << R"(
    }
  }
  llvm_unreachable("bogosity detected in disassembler state machine!");
}

)";
}

/// Collects all HwModes referenced by the target for encoding purposes.
void DecoderEmitter::collectHwModesReferencedForEncodings(
    std::vector<unsigned> &HwModeIDs,
    NamespacesHwModesMap &NamespacesWithHwModes) const {
  SmallBitVector BV(CGH.getNumModeIds());
  for (const auto &MS : CGH.getHwModeSelects()) {
    for (auto [HwModeID, EncodingDef] : MS.second.Items) {
      if (EncodingDef->isSubClassOf("InstructionEncoding")) {
        StringRef DecoderNamespace =
            EncodingDef->getValueAsString("DecoderNamespace");
        NamespacesWithHwModes[DecoderNamespace].insert(HwModeID);
        BV.set(HwModeID);
      }
    }
  }
  // FIXME: Can't do `HwModeIDs.assign(BV.set_bits_begin(), BV.set_bits_end())`
  //   because const_set_bits_iterator_impl is not copy-assignable.
  //   This breaks some MacOS builds.
  llvm::copy(BV.set_bits(), std::back_inserter(HwModeIDs));
}

void DecoderEmitter::handleHwModesUnrelatedEncodings(
    unsigned EncodingID, ArrayRef<unsigned> HwModeIDs,
    NamespacesHwModesMap &NamespacesWithHwModes) {
  switch (DecoderEmitterSuppressDuplicates) {
  case SUPPRESSION_DISABLE: {
    for (unsigned HwModeID : HwModeIDs)
      EncodingIDsByHwMode[HwModeID].push_back(EncodingID);
    break;
  }
  case SUPPRESSION_LEVEL1: {
    StringRef DecoderNamespace = Encodings[EncodingID].getDecoderNamespace();
    auto It = NamespacesWithHwModes.find(DecoderNamespace);
    if (It != NamespacesWithHwModes.end()) {
      for (unsigned HwModeID : It->second)
        EncodingIDsByHwMode[HwModeID].push_back(EncodingID);
    } else {
      // Only emit the encoding once, as it's DecoderNamespace doesn't
      // contain any HwModes.
      EncodingIDsByHwMode[DefaultMode].push_back(EncodingID);
    }
    break;
  }
  case SUPPRESSION_LEVEL2:
    EncodingIDsByHwMode[DefaultMode].push_back(EncodingID);
    break;
  }
}

/// Checks if the given target-specific non-pseudo instruction
/// is a candidate for decoding.
static bool isDecodableInstruction(const Record *InstDef) {
  return !InstDef->getValueAsBit("isAsmParserOnly") &&
         !InstDef->getValueAsBit("isCodeGenOnly");
}

/// Checks if the given encoding is valid.
static bool isValidEncoding(const Record *EncodingDef) {
  const RecordVal *InstField = EncodingDef->getValue("Inst");
  if (!InstField)
    return false;

  if (const auto *InstInit = dyn_cast<BitsInit>(InstField->getValue())) {
    // Fixed-length encoding. Size must be non-zero.
    if (!EncodingDef->getValueAsInt("Size"))
      return false;

    // At least one of the encoding bits must be complete (not '?').
    // FIXME: This should take SoftFail field into account.
    return !InstInit->allInComplete();
  }

  if (const auto *InstInit = dyn_cast<DagInit>(InstField->getValue())) {
    // Variable-length encoding.
    // At least one of the encoding bits must be complete (not '?').
    VarLenInst VLI(InstInit, InstField);
    return !all_of(VLI, [](const EncodingSegment &Segment) {
      return isa<UnsetInit>(Segment.Value);
    });
  }

  // Inst field is neither BitsInit nor DagInit. This is something unsupported.
  return false;
}

/// Parses all InstructionEncoding instances and fills internal data structures.
void DecoderEmitter::parseInstructionEncodings() {
  // First, collect all encoding-related HwModes referenced by the target.
  // And establish a mapping table between DecoderNamespace and HwMode.
  // If HwModeNames is empty, add the default mode so we always have one HwMode.
  std::vector<unsigned> HwModeIDs;
  NamespacesHwModesMap NamespacesWithHwModes;
  collectHwModesReferencedForEncodings(HwModeIDs, NamespacesWithHwModes);
  if (HwModeIDs.empty())
    HwModeIDs.push_back(DefaultMode);

  ArrayRef<const CodeGenInstruction *> Instructions =
      Target.getTargetNonPseudoInstructions();
  Encodings.reserve(Instructions.size());

  for (const CodeGenInstruction *Inst : Instructions) {
    const Record *InstDef = Inst->TheDef;
    if (!isDecodableInstruction(InstDef)) {
      ++NumEncodingsLackingDisasm;
      continue;
    }

    if (const Record *RV = InstDef->getValueAsOptionalDef("EncodingInfos")) {
      EncodingInfoByHwMode EBM(RV, CGH);
      for (auto [HwModeID, EncodingDef] : EBM) {
        if (!isValidEncoding(EncodingDef)) {
          // TODO: Should probably give a warning.
          ++NumEncodingsOmitted;
          continue;
        }
        unsigned EncodingID = Encodings.size();
        Encodings.emplace_back(EncodingDef, Inst);
        EncodingIDsByHwMode[HwModeID].push_back(EncodingID);
      }
      continue; // Ignore encoding specified by Instruction itself.
    }

    if (!isValidEncoding(InstDef)) {
      ++NumEncodingsOmitted;
      continue;
    }

    unsigned EncodingID = Encodings.size();
    Encodings.emplace_back(InstDef, Inst);

    // This instruction is encoded the same on all HwModes.
    // According to user needs, add it to all, some, or only the default HwMode.
    handleHwModesUnrelatedEncodings(EncodingID, HwModeIDs,
                                    NamespacesWithHwModes);
  }

  for (const Record *EncodingDef :
       RK.getAllDerivedDefinitions("AdditionalEncoding")) {
    const Record *InstDef = EncodingDef->getValueAsDef("AliasOf");
    // TODO: Should probably give a warning in these cases.
    //   What's the point of specifying an additional encoding
    //   if it is invalid or if the instruction is not decodable?
    if (!isDecodableInstruction(InstDef)) {
      ++NumEncodingsLackingDisasm;
      continue;
    }
    if (!isValidEncoding(EncodingDef)) {
      ++NumEncodingsOmitted;
      continue;
    }
    unsigned EncodingID = Encodings.size();
    Encodings.emplace_back(EncodingDef, &Target.getInstruction(InstDef));
    EncodingIDsByHwMode[DefaultMode].push_back(EncodingID);
  }

  // Do some statistics.
  NumInstructions = Instructions.size();
  NumEncodingsSupported = Encodings.size();
  NumEncodings = NumEncodingsSupported + NumEncodingsOmitted;
}

DecoderEmitter::DecoderEmitter(const RecordKeeper &RK)
    : RK(RK), Target(RK), CGH(Target.getHwModes()) {
  Target.reverseBitsForLittleEndianEncoding();
  parseInstructionEncodings();
}

// Emits disassembler code for instruction decoding.
void DecoderEmitter::run(raw_ostream &o) const {
  formatted_raw_ostream OS(o);
  OS << R"(
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include <assert.h>

namespace {

// InsnBitWidth is essentially a type trait used by the decoder emitter to query
// the supported bitwidth for a given type. But default, the value is 0, making
// it an invalid type for use as `InsnType` when instantiating the decoder.
// Individual targets are expected to provide specializations for these based
// on their usage.
template <typename T> constexpr uint32_t InsnBitWidth = 0;

)";

  // Do extra bookkeeping for variable-length encodings.
  bool IsVarLenInst = Target.hasVariableLengthEncodings();
  unsigned MaxInstLen = 0;
  if (IsVarLenInst) {
    std::vector<unsigned> InstrLen(Target.getInstructions().size(), 0);
    for (const InstructionEncoding &Encoding : Encodings) {
      MaxInstLen = std::max(MaxInstLen, Encoding.getBitWidth());
      InstrLen[Target.getInstrIntValue(Encoding.getInstruction()->TheDef)] =
          Encoding.getBitWidth();
    }

    // For variable instruction, we emit an instruction length table to let the
    // decoder know how long the instructions are. You can see example usage in
    // M68k's disassembler.
    emitInstrLenTable(OS, InstrLen);
  }

  // Map of (bitwidth, namespace, hwmode) tuple to encoding IDs.
  // Its organized as a nested map, with the (namespace, hwmode) as the key for
  // the inner map and bitwidth as the key for the outer map. We use std::map
  // for deterministic iteration order so that the code emitted is also
  // deterministic.
  using InnerKeyTy = std::pair<StringRef, unsigned>;
  using InnerMapTy = std::map<InnerKeyTy, std::vector<unsigned>>;
  std::map<unsigned, InnerMapTy> EncMap;

  for (const auto &[HwModeID, EncodingIDs] : EncodingIDsByHwMode) {
    for (unsigned EncodingID : EncodingIDs) {
      const InstructionEncoding &Encoding = Encodings[EncodingID];
      const unsigned BitWidth =
          IsVarLenInst ? MaxInstLen : Encoding.getBitWidth();
      StringRef DecoderNamespace = Encoding.getDecoderNamespace();
      EncMap[BitWidth][{DecoderNamespace, HwModeID}].push_back(EncodingID);
    }
  }

  // Variable length instructions use the same `APInt` type for all instructions
  // so we cannot specialize decoders based on instruction bitwidths (which
  // requires using different `InstType` for differet bitwidths for the correct
  // template specialization to kick in).
  if (IsVarLenInst && SpecializeDecodersPerBitwidth)
    PrintFatalError(
        "Cannot specialize decoders for variable length instuctions");

  // Entries in `EncMap` are already sorted by bitwidth. So bucketing per
  // bitwidth can be done on-the-fly as we iterate over the map.
  DecoderTableInfo TableInfo{};

  bool HasConflict = false;
  for (const auto &[BitWidth, BWMap] : EncMap) {
    for (const auto &[Key, EncodingIDs] : BWMap) {
      auto [DecoderNamespace, HwModeID] = Key;

      // Emit the decoder for this (namespace, hwmode, width) combination.
      std::unique_ptr<DecoderTreeNode> Tree =
          buildDecoderTree(Target, Encodings, EncodingIDs);
      if (!Tree) {
        HasConflict = true;
        continue;
      }

      SmallString<32> TableName("DecoderTable");
      TableName.append(DecoderNamespace);
      if (HwModeID != DefaultMode)
        TableName.append({"_", Target.getHwModes().getModeName(HwModeID)});
      TableName.append(std::to_string(BitWidth));

      // Serialize the tree.
      emitDecoderTable(OS, TableInfo, TableName, BitWidth, Tree.get());
    }

    // Each BitWidth get's its own decoders and decoder function if
    // SpecializeDecodersPerBitwidth is enabled.
    if (SpecializeDecodersPerBitwidth) {
      emitDecoderFunction(OS, TableInfo.Decoders, BitWidth);
      TableInfo.Decoders.clear();
    }
  }

  if (HasConflict)
    PrintFatalError("Decoding conflict encountered");

  // Emit the decoder function for the last bucket. This will also emit the
  // single decoder function if SpecializeDecodersPerBitwidth = false.
  if (!SpecializeDecodersPerBitwidth)
    emitDecoderFunction(OS, TableInfo.Decoders, 0);

  // Emit the predicate function.
  if (TableInfo.HasCheckPredicate)
    emitPredicateFunction(OS, TableInfo.Predicates);

  // Emit the main entry point for the decoder, decodeInstruction().
  emitDecodeInstruction(OS, IsVarLenInst, TableInfo);

  OS << "\n} // namespace\n";
}

void llvm::EmitDecoder(const RecordKeeper &RK, raw_ostream &OS) {
  DecoderEmitter(RK).run(OS);
}
