//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SparcSelectionDAGInfo.h"

#define GET_SDNODE_DESC
#include "SparcGenSDNodeInfo.inc"

using namespace llvm;

SparcSelectionDAGInfo::SparcSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(SparcGenSDNodeInfo) {}

void SparcSelectionDAGInfo::verifyTargetNode(const SelectionDAG &DAG,
                                             const SDNode *N) const {
  switch (N->getOpcode()) {
  case SPISD::CALL:
  case SPISD::TAIL_CALL:
  case SPISD::TLS_CALL:
    // operand #1 must have type i32, but has type i64
    return;
  }

  SelectionDAGGenTargetInfo::verifyTargetNode(DAG, N);
}

SparcSelectionDAGInfo::~SparcSelectionDAGInfo() = default;
