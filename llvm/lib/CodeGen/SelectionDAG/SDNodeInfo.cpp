//==------------------------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SDNodeInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLowering.h"

using namespace llvm;

static void reportNodeError(const SelectionDAG &DAG, const SDNode *N,
                            const Twine &Msg) {
  std::string S;
  raw_string_ostream SS(S);
  SS << "invalid node: " << Msg << '\n';
  N->printrWithDepth(SS, &DAG, 2);
  report_fatal_error(StringRef(S));
}

static void checkResultType(const SelectionDAG &DAG, const SDNode *N,
                            unsigned ResIdx, EVT ExpectedVT) {
  EVT ActualVT = N->getValueType(ResIdx);
  if (ActualVT != ExpectedVT)
    reportNodeError(
        DAG, N,
        "result #" + Twine(ResIdx) + " has invalid type; expected " +
            ExpectedVT.getEVTString() + ", got " + ActualVT.getEVTString());
}

static void checkOperandType(const SelectionDAG &DAG, const SDNode *N,
                             unsigned OpIdx, EVT ExpectedVT) {
  EVT ActualVT = N->getOperand(OpIdx).getValueType();
  if (ActualVT != ExpectedVT)
    reportNodeError(
        DAG, N,
        "operand #" + Twine(OpIdx) + " has invalid type; expected " +
            ExpectedVT.getEVTString() + ", got " + ActualVT.getEVTString());
}

namespace {

struct ConstraintOp {
  const SDNode *N;
  unsigned Idx;
  bool IsRes;

  SDValue getValue() const {
    return IsRes ? SDValue(const_cast<SDNode *>(N), Idx) : N->getOperand(Idx);
  }

  EVT getValueType() const { return getValue().getValueType(); }
};

raw_ostream &operator<<(raw_ostream &OS, const ConstraintOp &Info) {
  return OS << (Info.IsRes ? "result" : "operand") << " #" << Info.Idx;
}

} // namespace

void SDNodeInfo::verifyNode(const SelectionDAG &DAG, const SDNode *N) const {
  const SDNodeDesc &Desc = getDesc(N->getOpcode());
  bool HasChain = Desc.hasProperty(SDNPHasChain);
  bool HasOutGlue = Desc.hasProperty(SDNPOutGlue);
  bool HasInGlue = Desc.hasProperty(SDNPInGlue);
  bool HasOptInGlue = Desc.hasProperty(SDNPOptInGlue);
  bool IsVariadic = Desc.hasProperty(SDNPVariadic);

  unsigned ActualNumResults = N->getNumValues();
  unsigned ExpectedNumResults = Desc.NumResults + HasChain + HasOutGlue;

  if (ActualNumResults != ExpectedNumResults)
    reportNodeError(DAG, N,
                    "invalid number of results; expected " +
                        Twine(ExpectedNumResults) + ", got " +
                        Twine(ActualNumResults));

  // Chain result comes after all normal results.
  if (HasChain) {
    unsigned ChainResIdx = Desc.NumResults;
    checkResultType(DAG, N, ChainResIdx, MVT::Other);
  }

  // Glue result comes last.
  if (HasOutGlue) {
    unsigned GlueResIdx = Desc.NumResults + HasChain;
    checkResultType(DAG, N, GlueResIdx, MVT::Glue);
  }

  // In the most general case, the operands of a node go in the following order:
  //   chain, fix#0, ..., fix#M-1, var#0, ... var#N-1, glue
  // If the number of operands is < 0, M can be any;
  // If the node has SDNPVariadic property, N can be any.
  bool HasOptionalOperands = Desc.NumOperands < 0 || IsVariadic;

  unsigned ActualNumOperands = N->getNumOperands();
  unsigned ExpectedMinNumOperands =
      (Desc.NumOperands >= 0 ? Desc.NumOperands : 0) + HasChain + HasInGlue;

  // Check the lower bound.
  if (ActualNumOperands < ExpectedMinNumOperands) {
    StringRef How = HasOptionalOperands ? "at least " : "";
    reportNodeError(DAG, N,
                    "invalid number of operands; expected " + How +
                        Twine(ExpectedMinNumOperands) + ", got " +
                        Twine(ActualNumOperands));
  }

  // Check the upper bound. We can only do this if the number of fixed operands
  // is known and there are no variadic operands.
  if (Desc.NumOperands >= 0 && !IsVariadic) {
    // Account for optional input glue.
    unsigned ExpectedMaxNumOperands = ExpectedMinNumOperands + HasOptInGlue;
    if (ActualNumOperands > ExpectedMaxNumOperands) {
      StringRef How = HasOptInGlue ? "at most " : "";
      reportNodeError(DAG, N,
                      "invalid number of operands; expected " + How +
                          Twine(ExpectedMaxNumOperands) + ", got " +
                          Twine(ActualNumOperands));
    }
  }

  // Chain operand comes first.
  if (HasChain)
    checkOperandType(DAG, N, 0, MVT::Other);

  // Glue operand comes last.
  if (HasInGlue)
    checkOperandType(DAG, N, ActualNumOperands - 1, MVT::Glue);
  if (HasOptInGlue && ActualNumOperands >= 1 &&
      N->getOperand(ActualNumOperands - 1).getValueType() == MVT::Glue)
    HasInGlue = true;

  // Check variadic operands. These should be Register or RegisterMask.
  if (IsVariadic && Desc.NumOperands >= 0) {
    unsigned VarOpStart = HasChain + Desc.NumOperands;
    unsigned VarOpEnd = ActualNumOperands - HasInGlue;
    for (unsigned OpIdx = VarOpStart; OpIdx != VarOpEnd; ++OpIdx) {
      unsigned OpOpcode = N->getOperand(OpIdx).getOpcode();
      if (OpOpcode != ISD::Register && OpOpcode != ISD::RegisterMask)
        reportNodeError(DAG, N,
                        "variadic operand #" + Twine(OpIdx) +
                            " must be Register or RegisterMask");
    }
  }

  unsigned VTHwMode =
      DAG.getSubtarget().getHwMode(MCSubtargetInfo::HwMode_ValueType);

  auto GetConstraintVT = [&](const SDTypeConstraint &C) {
    if (C.NumHwModes == 0)
      return static_cast<MVT::SimpleValueType>(C.VT);
    for (auto [Mode, VT] : ArrayRef(&this->VTs[C.VT], C.NumHwModes)) {
      if (Mode == VTHwMode)
        return VT;
    }
    llvm_unreachable("HwMode not found");
  };

  auto GetConstraintOp = [&](unsigned Idx) {
    if (Idx < Desc.NumResults)
      return ConstraintOp{N, Idx, /*IsRes=*/true};
    return ConstraintOp{N, HasChain + (Idx - Desc.NumResults), /*IsRes=*/false};
  };

  std::string Errors;

  for (const SDTypeConstraint &C : getConstraints(N->getOpcode())) {
    SmallString<128> ES;
    raw_svector_ostream SS(ES);

    ConstraintOp Op = GetConstraintOp(C.OpNo);
    EVT OpVT = Op.getValue().getValueType();

    switch (C.Kind) {
    case SDTCisVT: {
      EVT ExpectedVT = GetConstraintVT(C);

      bool IsPtr = ExpectedVT == MVT::iPTR;
      if (IsPtr)
        ExpectedVT =
            DAG.getTargetLoweringInfo().getPointerTy(DAG.getDataLayout());

      if (OpVT != ExpectedVT) {
        SS << Op << " must have type " << ExpectedVT;
        if (IsPtr)
          SS << " (iPTR)";
        SS << ", but has type " << OpVT;
      }

      break;
    }
    case SDTCisPtrTy: {
      EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy(DAG.getDataLayout());
      if (OpVT != PtrVT) {
        SS << Op << " must have type " << PtrVT << " (iPTR), but has type "
           << OpVT;
      }
      break;
    }
    case SDTCisInt:
      if (!OpVT.isInteger()) {
        SS << Op << " must have integer type, but has type " << OpVT;
      }
      break;
    case SDTCisFP:
      if (!OpVT.isFloatingPoint()) {
        SS << Op << " must have floating-point type, but has type " << OpVT;
      }
      break;
    case SDTCisVec:
      if (!OpVT.isVector()) {
        SS << Op << " must have vector type, but has type " << OpVT;
      }
      break;
    case SDTCisSameAs: {
      ConstraintOp OtherOp = GetConstraintOp(C.OtherOpNo);
      EVT OtherOpVT = OtherOp.getValueType();

      if (OpVT != OtherOpVT) {
        SS << Op << " must have type " << OtherOpVT << " (same as " << OtherOp
           << "), but has type " << OpVT;
      }
      break;
    }
    case SDTCisVTSmallerThanOp:
      break;
    case SDTCisOpSmallerThanOp:
      break;
    case SDTCisEltOfVec: {
      ConstraintOp OtherOp = GetConstraintOp(C.OtherOpNo);
      EVT OtherOpVT = OtherOp.getValueType();

      if (OpVT.isVector()) {
        SS << Op << " must have scalar type";
      } else if (!OtherOpVT.isVector()) {
        SS << Op << " must have vector type";
      } else if (OpVT != OtherOpVT.getVectorElementType()) {
        SS << Op << " must have type " << OtherOpVT.getVectorElementType()
           << " (element of " << OtherOpVT << "), but has type " << OpVT;
      }
      break;
    }
    case SDTCisSubVecOfVec:
      break;
    case SDTCVecEltisVT: {
      EVT ExpectedVT = GetConstraintVT(C);
      assert(!ExpectedVT.isVector() &&
             (ExpectedVT.isInteger() || ExpectedVT.isFloatingPoint()));
      if (!OpVT.isVector()) {
        SS << Op << " must have vector type";
        reportNodeError(DAG, N, SS.str());
      } else if (OpVT.getVectorElementType() != ExpectedVT) {
        SS << Op << " must have " << ExpectedVT << " element type, but has "
           << OpVT.getVectorElementType() << " element type";
      }
      break;
    }
    case SDTCisSameNumEltsAs: {
      ConstraintOp OtherOp = GetConstraintOp(C.OtherOpNo);
      EVT OtherOpVT = OtherOp.getValueType();

      if (!OtherOpVT.isVector()) {
        if (OpVT.isVector()) {
          SS << Op << " must have scalar type since " << OtherOp
             << " has scalar type";
        }
      } else if (!OpVT.isVector()) {
        SS << Op << " must have vector type since " << OtherOp
           << " has vector type";
      } else if (OpVT.getVectorElementCount() !=
                 OtherOpVT.getVectorElementCount()) {
        SS << Op << " must have " << OtherOpVT.getVectorElementCount()
           << " elements (same as " << OtherOp << "), but has "
           << OpVT.getVectorElementCount() << " elements";
      }
      break;
    }
    case SDTCisSameSizeAs: {
      ConstraintOp OtherOp = GetConstraintOp(C.OtherOpNo);
      EVT OtherOpVT = OtherOp.getValueType();

      if (OpVT.getSizeInBits() != OtherOpVT.getSizeInBits()) {
        SS << Op << " must have size " << OtherOpVT.getSizeInBits()
           << " bits (same as " << OtherOp << "), but has size "
           << OpVT.getSizeInBits() << " bits";
      }
      break;
    }
    }

    if (!ES.empty()) {
      Errors.append("\n  ");
      Errors.append(std::string(ES));
    }
  }

  if (!Errors.empty())
    reportNodeError(DAG, N, "violated constraints:" + Errors);
}
