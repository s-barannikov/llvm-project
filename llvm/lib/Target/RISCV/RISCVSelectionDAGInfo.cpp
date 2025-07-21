//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVSelectionDAGInfo.h"

#define GET_SDNODE_DESC
#include "RISCVGenSDNodeInfo.inc"

using namespace llvm;

RISCVSelectionDAGInfo::RISCVSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(RISCVGenSDNodeInfo) {}

RISCVSelectionDAGInfo::~RISCVSelectionDAGInfo() = default;

void RISCVSelectionDAGInfo::verifyTargetNode(const SelectionDAG &DAG,
                                             const SDNode *N) const {
  switch (N->getOpcode()) {
  case RISCVISD::TUPLE_INSERT:
    // operand #1 must have vector type, but has type riscv_nxv16i8xN
  case RISCVISD::SETCC_VL:
    // result #0 must have type nxv1fN (same as operand #3),
    // but has type nxv1i1
  case RISCVISD::VSLIDEUP_VL:
    // result #0 must have vscale x M elements (same as operand #3),
    // but has vscale x N elements
  case RISCVISD::VSLIDE1DOWN_VL:
    // operand #1 must have type nxv2i32 (same as result #0),
    // but has type nxv1i64
    // operand #0 must have type nxv2i32 (same as result #0),
    // but has type nxv1i64
  case RISCVISD::VMV_V_X_VL:
    // result #0 must have type nxv2bf16 (same as operand #0),
    // but has type nxv2i16
  case RISCVISD::VCPOP_VL:
    // operand #0 must have M elements (same as operand #1),
    // but has vscale x N elements
  case RISCVISD::VECREDUCE_AND_VL:
  case RISCVISD::VECREDUCE_OR_VL:
  case RISCVISD::VECREDUCE_XOR_VL:
  case RISCVISD::VECREDUCE_FMAX_VL:
  case RISCVISD::VECREDUCE_FMIN_VL:
  case RISCVISD::VECREDUCE_SMAX_VL:
  case RISCVISD::VECREDUCE_SMIN_VL:
  case RISCVISD::VECREDUCE_UMAX_VL:
  case RISCVISD::VECREDUCE_UMIN_VL:
  case RISCVISD::VECREDUCE_ADD_VL:
  case RISCVISD::VECREDUCE_FADD_VL:
  case RISCVISD::VECREDUCE_SEQ_FADD_VL:
    // operand #1 must have N elements (same as operand #3),
    // but has vscale x M elements
    return;
  }

  SelectionDAGGenTargetInfo::verifyTargetNode(DAG, N);

#ifndef NDEBUG
  switch (N->getOpcode()) {
  case RISCVISD::TUPLE_EXTRACT:
    assert(N->getOperand(1).getOpcode() == ISD::TargetConstant &&
           "Expected index to be a target constant!");
    break;
  case RISCVISD::TUPLE_INSERT:
    assert(N->getOperand(2).getOpcode() == ISD::TargetConstant &&
           "Expected index to be a target constant!");
    break;
  case RISCVISD::VQDOT_VL:
  case RISCVISD::VQDOTU_VL:
  case RISCVISD::VQDOTSU_VL: {
    EVT VT = N->getValueType(0);
    assert(VT.isScalableVector() && VT.getVectorElementType() == MVT::i32 &&
           "Expected result to be an i32 scalable vector");
    assert((N->getOperand(4).getValueType() == MVT::i32 ||
            N->getOperand(4).getValueType() == MVT::i64) &&
           "Expect VL operand to be i32 or i64");
    break;
  }
  }
#endif
}
