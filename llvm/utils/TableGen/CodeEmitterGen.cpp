//===- CodeEmitterGen.cpp - Code Emitter Generator ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// CodeEmitterGen uses the descriptions of instructions and their fields to
// construct an automated code emitter: a function called
// getBinaryCodeForInstr() that, given a MCInst, returns the value of the
// instruction - either as an uint64_t or as an APInt, depending on the
// maximum bit width of all Inst definitions.
//
// In addition, it generates another function called getOperandBitOffset()
// that, given a MCInst and an operand index, returns the minimum of indices of
// all bits that carry some portion of the respective operand. When the target's
// encodeInstruction() stores the instruction in a little-endian byte order, the
// returned value is the offset of the start of the operand in the encoded
// instruction. Other targets might need to adjust the returned value according
// to their encodeInstruction() implementation.
//
//===----------------------------------------------------------------------===//

#include "Common/CodeGenHwModes.h"
#include "Common/CodeGenInstruction.h"
#include "Common/CodeGenTarget.h"
#include "Common/InfoByHwMode.h"
#include "Common/InstructionEncoding.h"
#include "Common/VarLenCodeEmitterGen.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

namespace {

class CodeEmitterGen {
  const RecordKeeper &RK;
  CodeGenTarget Target;
  const CodeGenHwModes &CGH;

public:
  explicit CodeEmitterGen(const RecordKeeper &RK);

  void run(raw_ostream &O);

private:
  int getVariableBit(const std::string &VarName, const BitsInit *BI, int Bit);
  std::pair<std::string, std::string> getInstructionCases(
      const CodeGenInstruction *Inst,
      const std::map<unsigned, DenseMap<unsigned, InstructionEncoding>>
          &EncodingsByHwMode);
  void addInstructionCasesForEncoding(const InstructionEncoding &Encoding,
                                      std::string &Case,
                                      std::string &BitOffsetCase);
  void addCodeToMergeInOperand(const OperandInfo &Op, std::string &Case,
                               std::string &BitOffsetCase);

  void emitInstructionBaseValues(
      raw_ostream &O, ArrayRef<const CodeGenInstruction *> NumberedInstructions,
      unsigned HwMode,
      const DenseMap<unsigned, InstructionEncoding> &Encodings);
  void
  emitCaseMap(raw_ostream &O,
              const std::map<std::string, std::vector<std::string>> &CaseMap);
  unsigned BitWidth = 0u;
  bool UseAPInt = false;
};

} // end anonymous namespace

// If the VarBitInit at position 'bit' matches the specified variable then
// return the variable bit position.  Otherwise return -1.
int CodeEmitterGen::getVariableBit(const std::string &VarName,
                                   const BitsInit *BI, int Bit) {
  if (const VarBitInit *VBI = dyn_cast<VarBitInit>(BI->getBit(Bit))) {
    if (const VarInit *VI = dyn_cast<VarInit>(VBI->getBitVar()))
      if (VI->getName() == VarName)
        return VBI->getBitNum();
  } else if (const VarInit *VI = dyn_cast<VarInit>(BI->getBit(Bit))) {
    if (VI->getName() == VarName)
      return 0;
  }

  return -1;
}

// Returns true if it succeeds, false if an error.
void CodeEmitterGen::addCodeToMergeInOperand(const OperandInfo &Op,
                                             std::string &Case,
                                             std::string &BitOffsetCase) {
  StringRef EncoderMethodName = Op.EncoderMethod;

  if (UseAPInt)
    Case += "      op.clearAllBits();\n";

  Case += "      // op: " + Op.Name.str() + "\n";

  // If the source operand has a custom encoder, use it.
  unsigned OpIdx = Op.OperandIndex;
  if (!EncoderMethodName.empty()) {
    raw_string_ostream CaseOS(Case);
    CaseOS << indent(6);
    if (UseAPInt)
      CaseOS << EncoderMethodName << "(MI, " << OpIdx << ", op";
    else
      CaseOS << "op = " << EncoderMethodName << "(MI, " << OpIdx;
    CaseOS << ", Fixups, STI);\n";
  } else {
    if (UseAPInt) {
      Case +=
          "      getMachineOpValue(MI, MI.getOperand(" + utostr(OpIdx) + ")";
      Case += ", op, Fixups, STI";
    } else {
      Case += "      op = getMachineOpValue(MI, MI.getOperand(" +
              utostr(OpIdx) + ")";
      Case += ", Fixups, STI";
    }
    Case += ");\n";
  }

  for (const EncodingField &Field : Op.fields()) {
    std::string MaskStr;
    int OpShift;

    unsigned LoBit = Field.Offset;
    unsigned HiBit = LoBit + Field.Width;
    unsigned LoInstBit = Field.Base;
    if (UseAPInt) {
      std::string ExtractStr;
      if (Field.Width >= 64) {
        ExtractStr = "op.extractBits(" + itostr(HiBit - LoBit) + ", " +
                     itostr(LoBit) + ")";
        Case += "      Value.insertBits(" + ExtractStr + ", " +
                itostr(LoInstBit) + ");\n";
      } else {
        ExtractStr = "op.extractBitsAsZExtValue(" + itostr(HiBit - LoBit) +
                     ", " + itostr(LoBit) + ")";
        Case += "      Value.insertBits(" + ExtractStr + ", " +
                itostr(LoInstBit) + ", " + itostr(HiBit - LoBit) + ");\n";
      }
    } else {
      uint64_t OpMask = ~(uint64_t)0 >> (64 - Field.Width);
      OpShift = LoBit;
      OpMask <<= OpShift;
      MaskStr = "UINT64_C(" + utostr(OpMask) + ")";
      OpShift = LoInstBit - LoBit;

      if (Op.fields().size() == 1) {
        Case += "      op &= " + MaskStr + ";\n";
        if (OpShift > 0) {
          Case += "      op <<= " + itostr(OpShift) + ";\n";
        } else if (OpShift < 0) {
          Case += "      op >>= " + itostr(-OpShift) + ";\n";
        }
        Case += "      Value |= op;\n";
      } else {
        if (OpShift > 0) {
          Case += "      Value |= (op & " + MaskStr + ") << " +
                  itostr(OpShift) + ";\n";
        } else if (OpShift < 0) {
          Case += "      Value |= (op & " + MaskStr + ") >> " +
                  itostr(-OpShift) + ";\n";
        } else {
          Case += "      Value |= (op & " + MaskStr + ");\n";
        }
      }
    }
  }

  auto LessFieldByBase = [](const EncodingField &A, const EncodingField &B) {
    return A.Base < B.Base;
  };
  auto I = min_element(Op.fields(), LessFieldByBase);
  BitOffsetCase += "      case " + utostr(OpIdx) + ":\n";
  BitOffsetCase += "        // op: " + Op.Name.str() + "\n";
  BitOffsetCase += "        return " + utostr(I->Base) + ";\n";
}

std::pair<std::string, std::string> CodeEmitterGen::getInstructionCases(
    const CodeGenInstruction *Inst,
    const std::map<unsigned, DenseMap<unsigned, InstructionEncoding>>
        &EncodingsByHwMode) {
  std::string Case, BitOffsetCase;

  auto Append = [&](const std::string &S) {
    Case += S;
    BitOffsetCase += S;
  };

  SmallVector<std::pair<unsigned, const InstructionEncoding *>, 4>
      InstEncodings;
  for (const auto &[HwMode, Encodings] : EncodingsByHwMode)
    if (auto I = Encodings.find(Inst->EnumVal); I != Encodings.end())
      InstEncodings.emplace_back(HwMode, &I->second);

  if (InstEncodings.empty())
    return {std::move(Case), std::move(BitOffsetCase)};

  if (InstEncodings.size() == 1 && InstEncodings.front().first == DefaultMode) {
    addInstructionCasesForEncoding(*InstEncodings.front().second, Case,
                                   BitOffsetCase);
    return {std::move(Case), std::move(BitOffsetCase)};
  }

  // Invoke the interface to obtain the HwMode ID controlling the
  // EncodingInfo for the current subtarget. This interface will
  // mask off irrelevant HwMode IDs.
  Append("      unsigned HwMode = "
         "STI.getHwMode(MCSubtargetInfo::HwMode_EncodingInfo);\n");
  Case += "      switch (HwMode) {\n";
  Case += "      default: llvm_unreachable(\"Unknown hardware mode!\"); "
          "break;\n";
  for (unsigned ModeId : make_first_range(InstEncodings)) {
    if (ModeId == DefaultMode) {
      Case += "      case " + itostr(DefaultMode) + ": InstBitsByHw = InstBits";
    } else {
      Case += "      case " + itostr(ModeId) + ": InstBitsByHw = InstBits_" +
              CGH.getMode(ModeId).Name.str();
    }
    Case += "; break;\n";
  }
  Case += "      };\n";

  // We need to remodify the 'Inst' value from the table we found above.
  if (UseAPInt) {
    int NumWords = APInt::getNumWords(BitWidth);
    Case += "      Inst = APInt(" + itostr(BitWidth);
    Case += ", ArrayRef(InstBitsByHw + TableIndex * " + itostr(NumWords) +
            ", " + itostr(NumWords);
    Case += "));\n";
    Case += "      Value = Inst;\n";
  } else {
    Case += "      Value = InstBitsByHw[TableIndex];\n";
  }

  Append("      switch (HwMode) {\n");
  Append("      default: llvm_unreachable(\"Unhandled HwMode\");\n");
  for (auto [ModeId, Encoding] : InstEncodings) {
    Append("      case " + itostr(ModeId) + ": {\n");
    addInstructionCasesForEncoding(*Encoding, Case, BitOffsetCase);
    Append("      break;\n");
    Append("      }\n");
  }
  Append("      }\n");
  return {std::move(Case), std::move(BitOffsetCase)};
}

void CodeEmitterGen::addInstructionCasesForEncoding(
    const InstructionEncoding &Encoding, std::string &Case,
    std::string &BitOffsetCase) {
  const Record *R = Encoding.getInstruction()->TheDef;

  // Loop over all of the fields in the instruction.
  size_t OrigBitOffsetCaseSize = BitOffsetCase.size();
  BitOffsetCase += "      switch (OpNum) {\n";
  size_t BitOffsetCaseSizeBeforeLoop = BitOffsetCase.size();
  for (const OperandInfo &Op : Encoding.getOperands())
    if (!Op.fields().empty())
      addCodeToMergeInOperand(Op, Case, BitOffsetCase);
  // Avoid empty switches.
  if (BitOffsetCase.size() == BitOffsetCaseSizeBeforeLoop)
    BitOffsetCase.resize(OrigBitOffsetCaseSize);
  else
    BitOffsetCase += "      }\n";

  StringRef PostEmitter = R->getValueAsString("PostEncoderMethod");
  if (!PostEmitter.empty()) {
    Case += "      Value = ";
    Case += PostEmitter;
    Case += "(MI, Value";
    Case += ", STI";
    Case += ");\n";
  }
}

static void emitInstBits(raw_ostream &OS, const APInt &Bits) {
  for (unsigned I = 0; I < Bits.getNumWords(); ++I)
    OS << ((I > 0) ? ", " : "") << "UINT64_C(" << Bits.getRawData()[I] << ")";
}

void CodeEmitterGen::emitInstructionBaseValues(
    raw_ostream &O, ArrayRef<const CodeGenInstruction *> NumberedInstructions,
    unsigned HwMode, const DenseMap<unsigned, InstructionEncoding> &Encodings) {
  if (HwMode == DefaultMode)
    O << "  static const uint64_t InstBits[] = {\n";
  else
    O << "  static const uint64_t InstBits_" << CGH.getModeName(HwMode)
      << "[] = {\n";

  for (const CodeGenInstruction *CGI : NumberedInstructions) {
    const Record *R = CGI->TheDef;
    auto I = Encodings.find(CGI->EnumVal);
    if (I == Encodings.end()) {
      // If the HwMode does not match, then Encoding '0'
      // should be generated.
      APInt Value(BitWidth, 0);
      O << "    ";
      emitInstBits(O, Value);
      O << "," << '\t' << "// " << R->getName() << "\n";
      continue;
    }

    // Start by filling in fixed values.
    APInt Value = I->second.getInstBits().One.zext(BitWidth);
    O << "    ";
    emitInstBits(O, Value);
    O << "," << '\t' << "// " << R->getName() << "\n";
  }
  O << "  };\n";
}

void CodeEmitterGen::emitCaseMap(
    raw_ostream &O,
    const std::map<std::string, std::vector<std::string>> &CaseMap) {
  for (const auto &[Case, InstList] : CaseMap) {
    bool First = true;
    for (const auto &Inst : InstList) {
      if (!First)
        O << "\n";
      O << "    case " << Inst << ":";
      First = false;
    }
    O << " {\n";
    O << Case;
    O << "      break;\n"
      << "    }\n";
  }
}

CodeEmitterGen::CodeEmitterGen(const RecordKeeper &RK)
    : RK(RK), Target(RK), CGH(Target.getHwModes()) {
  // For little-endian instruction bit encodings, reverse the bit order.
  Target.reverseBitsForLittleEndianEncoding();
}

void CodeEmitterGen::run(raw_ostream &O) {
  emitSourceFileHeader("Machine Code Emitter", O);

  ArrayRef<const CodeGenInstruction *> EncodedInstructions =
      Target.getTargetNonPseudoInstructions();

  if (Target.hasVariableLengthEncodings()) {
    emitVarLenCodeEmitter(RK, O);
    return;
  }

  // Map {HwMode: {Opcode: Encoding}}.
  std::map<unsigned, DenseMap<unsigned, InstructionEncoding>> EncodingsByHwMode;
  EncodingsByHwMode.try_emplace(DefaultMode);

  BitWidth = 0;
  for (const CodeGenInstruction *CGI : EncodedInstructions) {
    const Record *R = CGI->TheDef;
    if (const Record *RV = R->getValueAsOptionalDef("EncodingInfos")) {
      EncodingInfoByHwMode EBM(RV, CGH);
      for (const auto &[Key, Value] : EBM) {
        auto [I, Inserted] =
            EncodingsByHwMode[Key].try_emplace(CGI->EnumVal, Value, CGI);
        BitWidth = std::max(BitWidth, I->second.getBitWidth());
      }
      continue;
    }
    auto [I, Inserted] =
        EncodingsByHwMode[DefaultMode].try_emplace(CGI->EnumVal, R, CGI);
    BitWidth = std::max(BitWidth, I->second.getBitWidth());
  }

  // Emit function declaration
  if (UseAPInt) {
    O << "void " << Target.getName()
      << "MCCodeEmitter::getBinaryCodeForInstr(const MCInst &MI,\n"
      << "    SmallVectorImpl<MCFixup> &Fixups,\n"
      << "    APInt &Inst,\n"
      << "    APInt &Scratch,\n"
      << "    const MCSubtargetInfo &STI) const {\n";
  } else {
    O << "uint64_t " << Target.getName();
    O << "MCCodeEmitter::getBinaryCodeForInstr(const MCInst &MI,\n"
      << "    SmallVectorImpl<MCFixup> &Fixups,\n"
      << "    const MCSubtargetInfo &STI) const {\n";
  }

  // Emit instruction base values
  for (const auto &[HwMode, Encodings] : EncodingsByHwMode)
    emitInstructionBaseValues(O, EncodedInstructions, HwMode, Encodings);

  if (EncodingsByHwMode.size() > 1) {
    // This pointer will be assigned to the HwMode table later.
    O << "  const uint64_t *InstBitsByHw;\n";
  }

  // Map to accumulate all the cases.
  std::map<std::string, std::vector<std::string>> CaseMap;
  std::map<std::string, std::vector<std::string>> BitOffsetCaseMap;

  // Construct all cases statement for each opcode
  for (const CodeGenInstruction *CGI : EncodedInstructions) {
    const Record *R = CGI->TheDef;
    std::string InstName =
        (R->getValueAsString("Namespace") + "::" + R->getName()).str();
    std::string Case, BitOffsetCase;
    std::tie(Case, BitOffsetCase) = getInstructionCases(CGI, EncodingsByHwMode);

    CaseMap[Case].push_back(InstName);
    BitOffsetCaseMap[BitOffsetCase].push_back(std::move(InstName));
  }

  unsigned FirstSupportedOpcode = EncodedInstructions.front()->EnumVal;
  O << "  constexpr unsigned FirstSupportedOpcode = " << FirstSupportedOpcode
    << ";\n";
  O << R"(
  const unsigned opcode = MI.getOpcode();
  if (opcode < FirstSupportedOpcode) {
    std::string msg;
    raw_string_ostream Msg(msg);
    Msg << "Unsupported instruction: " << MI;
    report_fatal_error(Msg.str().c_str());
  }
  unsigned TableIndex = opcode - FirstSupportedOpcode;
)";

  // Emit initial function code
  if (UseAPInt) {
    int NumWords = APInt::getNumWords(BitWidth);
    O << "  if (Scratch.getBitWidth() != " << BitWidth << ")\n"
      << "    Scratch = Scratch.zext(" << BitWidth << ");\n"
      << "  Inst = APInt(" << BitWidth << ", ArrayRef(InstBits + TableIndex * "
      << NumWords << ", " << NumWords << "));\n"
      << "  APInt &Value = Inst;\n"
      << "  APInt &op = Scratch;\n"
      << "  switch (opcode) {\n";
  } else {
    O << "  uint64_t Value = InstBits[TableIndex];\n"
      << "  uint64_t op = 0;\n"
      << "  (void)op;  // suppress warning\n"
      << "  switch (opcode) {\n";
  }

  // Emit each case statement
  emitCaseMap(O, CaseMap);

  // Default case: unhandled opcode.
  O << "  default:\n"
    << "    std::string msg;\n"
    << "    raw_string_ostream Msg(msg);\n"
    << "    Msg << \"Not supported instr: \" << MI;\n"
    << "    report_fatal_error(Msg.str().c_str());\n"
    << "  }\n";
  if (UseAPInt)
    O << "  Inst = Value;\n";
  else
    O << "  return Value;\n";
  O << "}\n\n";

  O << "#ifdef GET_OPERAND_BIT_OFFSET\n"
    << "#undef GET_OPERAND_BIT_OFFSET\n\n"
    << "uint32_t " << Target.getName()
    << "MCCodeEmitter::getOperandBitOffset(const MCInst &MI,\n"
    << "    unsigned OpNum,\n"
    << "    const MCSubtargetInfo &STI) const {\n"
    << "  switch (MI.getOpcode()) {\n";
  emitCaseMap(O, BitOffsetCaseMap);
  O << "  }\n"
    << "  std::string msg;\n"
    << "  raw_string_ostream Msg(msg);\n"
    << "  Msg << \"Not supported instr[opcode]: \" << MI << \"[\" << OpNum "
       "<< \"]\";\n"
    << "  report_fatal_error(Msg.str().c_str());\n"
    << "}\n\n"
    << "#endif // GET_OPERAND_BIT_OFFSET\n\n";
}

static TableGen::Emitter::OptClass<CodeEmitterGen>
    X("gen-emitter", "Generate machine code emitter");
