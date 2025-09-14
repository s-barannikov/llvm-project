//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common/CodeGenTarget.h"
#include "Common/InstructionEncoding.h"
#include "Common/SubtargetFeatureInfo.h"
#include "DecoderTree.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"

namespace llvm {
extern cl::opt<bool> UseFnTableInDecodeToMCInst;
extern cl::opt<bool> IgnoreNonDecodableOperands;
extern cl::opt<bool> IgnoreFullyDefinedOperands;
} // namespace llvm

using namespace llvm;

namespace {

/// Filter - Filter works with FilterChooser to produce the decoding tree for
/// the ISA.
///
/// It is useful to think of a Filter as governing the switch stmts of the
/// decoding tree in a certain level.  Each case stmt delegates to an inferior
/// FilterChooser to decide what further decoding logic to employ, or in another
/// words, what other remaining bits to look at.  The FilterChooser eventually
/// chooses a best Filter to do its job.
///
/// This recursive scheme ends when the number of Opcodes assigned to the
/// FilterChooser becomes 1 or if there is a conflict.  A conflict happens when
/// the Filter/FilterChooser combo does not know how to distinguish among the
/// Opcodes assigned.
///
/// An example of a conflict is
///
/// Decoding Conflict:
///     ................................
///     1111............................
///     1111010.........................
///     1111010...00....................
///     1111010...00........0001........
///     111101000.00........0001........
///     111101000.00........00010000....
///     111101000_00________00010000____  VST4q8a
///     111101000_00________00010000____  VST4q8b
///
/// The Debug output shows the path that the decoding tree follows to reach the
/// the conclusion that there is a conflict.  VST4q8a is a vst4 to double-spaced
/// even registers, while VST4q8b is a vst4 to double-spaced odd registers.
///
/// The encoding info in the .td files does not specify this meta information,
/// which could have been used by the decoder to resolve the conflict.  The
/// decoder could try to decode the even/odd register numbering and assign to
/// VST4q8a or VST4q8b, but for the time being, the decoder chooses the "a"
/// version and return the Opcode since the two have the same Asm format string.
struct Filter {
  unsigned StartBit; // the starting bit position
  unsigned NumBits;  // number of bits to filter

  // Map of well-known segment value to the set of uid's with that value.
  std::map<uint64_t, std::vector<unsigned>> FilteredIDs;

  // Set of uid's with non-constant segment values.
  std::vector<unsigned> VariableIDs;

  Filter(ArrayRef<InstructionEncoding> Encodings,
         ArrayRef<unsigned> EncodingIDs, unsigned StartBit, unsigned NumBits);

  // Returns the number of fanout produced by the filter.  More fanout implies
  // the filter distinguishes more categories of instructions.
  unsigned usefulness() const {
    return FilteredIDs.size() + VariableIDs.empty();
  }
};

} // namespace

Filter::Filter(ArrayRef<InstructionEncoding> Encodings,
               ArrayRef<unsigned> EncodingIDs, unsigned StartBit,
               unsigned NumBits)
    : StartBit(StartBit), NumBits(NumBits) {
  for (unsigned EncodingID : EncodingIDs) {
    const InstructionEncoding &Encoding = Encodings[EncodingID];
    KnownBits EncodingBits = Encoding.getMandatoryBits();

    // Scans the segment for possibly well-specified encoding bits.
    KnownBits FieldBits = EncodingBits.extractBits(NumBits, StartBit);

    if (FieldBits.isConstant()) {
      // The encoding bits are well-known.  Lets add the uid of the
      // instruction into the bucket keyed off the constant field value.
      FilteredIDs[FieldBits.getConstant().getZExtValue()].push_back(EncodingID);
    } else {
      // Some of the encoding bit(s) are unspecified.  This contributes to
      // one additional member of "Variable" instructions.
      VariableIDs.push_back(EncodingID);
    }
  }

  assert((FilteredIDs.size() + VariableIDs.size() > 0) &&
         "Filter returns no instruction categories");
}

namespace {

// These are states of our finite state machines used in FilterChooser's
// filterProcessor() which produces the filter candidates to use.
enum bitAttr_t {
  ATTR_NONE,
  ATTR_FILTERED,
  ATTR_ALL_SET,
  ATTR_ALL_UNSET,
  ATTR_MIXED
};

/// FilterChooser - FilterChooser chooses the best filter among a set of Filters
/// in order to perform the decoding of instructions at the current level.
///
/// Decoding proceeds from the top down.  Based on the well-known encoding bits
/// of instructions available, FilterChooser builds up the possible Filters that
/// can further the task of decoding by distinguishing among the remaining
/// candidate instructions.
///
/// Once a filter has been chosen, it is called upon to divide the decoding task
/// into sub-tasks and delegates them to its inferior FilterChoosers for further
/// processings.
///
/// It is useful to think of a Filter as governing the switch stmts of the
/// decoding tree.  And each case is delegated to an inferior FilterChooser to
/// decide what further remaining bits to look at.

class FilterChooser {
  // TODO: Unfriend by providing the necessary accessors.
  friend class DecoderTreeBuilder;

  // Vector of encodings to choose our filter.
  ArrayRef<InstructionEncoding> Encodings;

  /// Encoding IDs for this filter chooser to work on.
  /// Sorted by non-decreasing encoding width.
  SmallVector<unsigned, 0> EncodingIDs;

  // Array of bit values passed down from our parent.
  // Set to all unknown for Parent == nullptr.
  KnownBits FilterBits;

  // Links to the FilterChooser above us in the decoding tree.
  const FilterChooser *Parent;

  /// If the selected filter matches multiple encodings, then this is the
  /// starting position and the width of the filtered range.
  unsigned StartBit;
  unsigned NumBits;

  /// If the selected filter matches multiple encodings, and there is
  /// *exactly one* encoding in which all bits are known in the filtered range,
  /// then this is the ID of that encoding.
  /// Also used when there is only one encoding.
  std::optional<unsigned> SingletonEncodingID;

  /// If the selected filter matches multiple encodings, and there is
  /// *at least one* encoding in which all bits are known in the filtered range,
  /// then this is the FilterChooser created for the subset of encodings that
  /// contain some unknown bits in the filtered range.
  std::unique_ptr<const FilterChooser> VariableFC;

  /// If the selected filter matches multiple encodings, and there is
  /// *more than one* encoding in which all bits are known in the filtered
  /// range, then this is a map of field values to FilterChoosers created for
  /// the subset of encodings sharing that field value.
  /// The "field value" here refers to the encoding bits in the filtered range.
  std::map<uint64_t, std::unique_ptr<const FilterChooser>> FilterChooserMap;

  /// Set to true if decoding conflict was encountered.
  bool HasConflict = false;

  struct Island {
    unsigned StartBit;
    unsigned NumBits;
    uint64_t FieldVal;
  };

public:
  /// Constructs a top-level filter chooser.
  FilterChooser(ArrayRef<InstructionEncoding> Encodings,
                ArrayRef<unsigned> EncodingIDs);

  /// Constructs an inferior filter chooser.
  FilterChooser(ArrayRef<InstructionEncoding> Encodings,
                ArrayRef<unsigned> EncodingIDs, const KnownBits &FilterBits,
                const FilterChooser &Parent);

  /// Returns the width of the largest encoding.
  unsigned getMaxEncodingWidth() const {
    // The last encoding ID is the ID of an encoding with the largest width.
    return Encodings[EncodingIDs.back()].getBitWidth();
  }

  /// Returns true if any decoding conflicts were encountered.
  bool hasConflict() const { return HasConflict; }

private:
  /// Applies the given filter to the set of encodings this FilterChooser
  /// works with, creating inferior FilterChoosers as necessary.
  void applyFilter(const Filter &F);

  /// dumpStack - dumpStack traverses the filter chooser chain and calls
  /// dumpFilterArray on each filter chooser up to the top level one.
  void dumpStack(raw_ostream &OS, indent Indent, unsigned PadToWidth) const;

  bool isPositionFiltered(unsigned Idx) const {
    return FilterBits.Zero[Idx] || FilterBits.One[Idx];
  }

  // Calculates the island(s) needed to decode the instruction.
  // This returns a list of undecoded bits of an instructions, for example,
  // Inst{20} = 1 && Inst{3-0} == 0b1111 represents two islands of yet-to-be
  // decoded bits in order to verify that the instruction matches the Opcode.
  static std::vector<Island> getIslands(const KnownBits &EncodingBits,
                                        const KnownBits &FilterBits);

  /// Scans the well-known encoding bits of the encodings and, builds up a list
  /// of candidate filters, and then returns the best one, if any.
  std::unique_ptr<Filter> findBestFilter(ArrayRef<bitAttr_t> BitAttrs,
                                         bool AllowMixed,
                                         bool Greedy = true) const;

  std::unique_ptr<Filter> findBestFilter() const;

  // Decides on the best configuration of filter(s) to use in order to decode
  // the instructions.  A conflict of instructions may occur, in which case we
  // dump the conflict set to the standard error.
  void doFilter();

public:
  void dump() const;
};

/// Sorting predicate to sort encoding IDs by encoding width.
class LessEncodingIDByWidth {
  ArrayRef<InstructionEncoding> Encodings;

public:
  explicit LessEncodingIDByWidth(ArrayRef<InstructionEncoding> Encodings)
      : Encodings(Encodings) {}

  bool operator()(unsigned ID1, unsigned ID2) const {
    return Encodings[ID1].getBitWidth() < Encodings[ID2].getBitWidth();
  }
};

} // namespace

FilterChooser::FilterChooser(ArrayRef<InstructionEncoding> Encodings,
                             ArrayRef<unsigned> EncodingIDs)
    : Encodings(Encodings), EncodingIDs(EncodingIDs), Parent(nullptr) {
  // Sort encoding IDs once.
  stable_sort(this->EncodingIDs, LessEncodingIDByWidth(Encodings));
  // Filter width is the width of the smallest encoding.
  unsigned FilterWidth = Encodings[this->EncodingIDs.front()].getBitWidth();
  FilterBits = KnownBits(FilterWidth);
  doFilter();
}

FilterChooser::FilterChooser(ArrayRef<InstructionEncoding> Encodings,
                             ArrayRef<unsigned int> EncodingIDs,
                             const KnownBits &FilterBits,
                             const FilterChooser &Parent)
    : Encodings(Encodings), EncodingIDs(EncodingIDs), Parent(&Parent) {
  // Inferior filter choosers are created from sorted array of encoding IDs.
  assert(is_sorted(EncodingIDs, LessEncodingIDByWidth(Encodings)));
  assert(!FilterBits.hasConflict() && "Broken filter");
  // Filter width is the width of the smallest encoding.
  unsigned FilterWidth = Encodings[EncodingIDs.front()].getBitWidth();
  this->FilterBits = FilterBits.anyext(FilterWidth);
  doFilter();
}

void FilterChooser::applyFilter(const Filter &F) {
  StartBit = F.StartBit;
  NumBits = F.NumBits;
  assert(FilterBits.extractBits(NumBits, StartBit).isUnknown());

  if (!F.VariableIDs.empty()) {
    // Delegates to an inferior filter chooser for further processing on this
    // group of instructions whose segment values are variable.
    VariableFC = std::make_unique<FilterChooser>(Encodings, F.VariableIDs,
                                                 FilterBits, *this);
    HasConflict |= VariableFC->HasConflict;
  }

  // Otherwise, create sub choosers.
  for (const auto &[FilterVal, InferiorEncodingIDs] : F.FilteredIDs) {
    // Create a new filter by inserting the field bits into the parent filter.
    APInt FieldBits(NumBits, FilterVal);
    KnownBits InferiorFilterBits = FilterBits;
    InferiorFilterBits.insertBits(KnownBits::makeConstant(FieldBits), StartBit);

    // Delegates to an inferior filter chooser for further processing on this
    // category of instructions.
    auto [It, _] = FilterChooserMap.try_emplace(
        FilterVal,
        std::make_unique<FilterChooser>(Encodings, InferiorEncodingIDs,
                                        InferiorFilterBits, *this));
    HasConflict |= It->second->HasConflict;
  }
}

/// Similar to KnownBits::print(), but allows you to specify a character to use
/// to print unknown bits.
static void printKnownBits(raw_ostream &OS, const KnownBits &Bits,
                           char Unknown) {
  for (unsigned I = Bits.getBitWidth(); I--;) {
    if (Bits.Zero[I] && Bits.One[I])
      OS << '!';
    else if (Bits.Zero[I])
      OS << '0';
    else if (Bits.One[I])
      OS << '1';
    else
      OS << Unknown;
  }
}

/// dumpStack - dumpStack traverses the filter chooser chain and calls
/// dumpFilterArray on each filter chooser up to the top level one.
void FilterChooser::dumpStack(raw_ostream &OS, indent Indent,
                              unsigned PadToWidth) const {
  if (Parent)
    Parent->dumpStack(OS, Indent, PadToWidth);
  assert(PadToWidth >= FilterBits.getBitWidth());
  OS << Indent << indent(PadToWidth - FilterBits.getBitWidth());
  printKnownBits(OS, FilterBits, '.');
  OS << '\n';
}

// Calculates the island(s) needed to decode the instruction.
// This returns a list of undecoded bits of an instructions, for example,
// Inst{20} = 1 && Inst{3-0} == 0b1111 represents two islands of yet-to-be
// decoded bits in order to verify that the instruction matches the Opcode.
std::vector<FilterChooser::Island>
FilterChooser::getIslands(const KnownBits &EncodingBits,
                          const KnownBits &FilterBits) {
  std::vector<Island> Islands;
  uint64_t FieldVal;
  unsigned StartBit;

  // 0: Init
  // 1: Water (the bit value does not affect decoding)
  // 2: Island (well-known bit value needed for decoding)
  unsigned State = 0;

  unsigned FilterWidth = FilterBits.getBitWidth();
  for (unsigned i = 0; i != FilterWidth; ++i) {
    bool IsKnown = EncodingBits.Zero[i] || EncodingBits.One[i];
    bool Filtered = FilterBits.Zero[i] || FilterBits.One[i];
    switch (State) {
    default:
      llvm_unreachable("Unreachable code!");
    case 0:
    case 1:
      if (Filtered || !IsKnown) {
        State = 1; // Still in Water
      } else {
        State = 2; // Into the Island
        StartBit = i;
        FieldVal = static_cast<uint64_t>(EncodingBits.One[i]);
      }
      break;
    case 2:
      if (Filtered || !IsKnown) {
        State = 1; // Into the Water
        Islands.push_back({StartBit, i - StartBit, FieldVal});
      } else {
        State = 2; // Still in Island
        FieldVal |= static_cast<uint64_t>(EncodingBits.One[i])
                    << (i - StartBit);
      }
      break;
    }
  }
  // If we are still in Island after the loop, do some housekeeping.
  if (State == 2)
    Islands.push_back({StartBit, FilterWidth - StartBit, FieldVal});

  return Islands;
}

std::unique_ptr<Filter>
FilterChooser::findBestFilter(ArrayRef<bitAttr_t> BitAttrs, bool AllowMixed,
                              bool Greedy) const {
  assert(EncodingIDs.size() >= 2 && "Nothing to filter");

  // Heuristics.  See also doFilter()'s "Heuristics" comment when num of
  // instructions is 3.
  if (AllowMixed && !Greedy) {
    assert(EncodingIDs.size() == 3);

    for (unsigned EncodingID : EncodingIDs) {
      const InstructionEncoding &Encoding = Encodings[EncodingID];
      KnownBits EncodingBits = Encoding.getMandatoryBits();

      // Look for islands of undecoded bits of any instruction.
      std::vector<Island> Islands = getIslands(EncodingBits, FilterBits);
      if (!Islands.empty()) {
        // Found an instruction with island(s).  Now just assign a filter.
        return std::make_unique<Filter>(
            Encodings, EncodingIDs, Islands[0].StartBit, Islands[0].NumBits);
      }
    }
  }

  // The regionAttr automaton consumes the bitAttrs automatons' state,
  // lowest-to-highest.
  //
  //   Input symbols: F(iltered), (all_)S(et), (all_)U(nset), M(ixed)
  //   States:        NONE, ALL_SET, MIXED
  //   Initial state: NONE
  //
  // (NONE) ----- F --> (NONE)
  // (NONE) ----- S --> (ALL_SET)     ; and set region start
  // (NONE) ----- U --> (NONE)
  // (NONE) ----- M --> (MIXED)       ; and set region start
  // (ALL_SET) -- F --> (NONE)        ; and report an ALL_SET region
  // (ALL_SET) -- S --> (ALL_SET)
  // (ALL_SET) -- U --> (NONE)        ; and report an ALL_SET region
  // (ALL_SET) -- M --> (MIXED)       ; and report an ALL_SET region
  // (MIXED) ---- F --> (NONE)        ; and report a MIXED region
  // (MIXED) ---- S --> (ALL_SET)     ; and report a MIXED region
  // (MIXED) ---- U --> (NONE)        ; and report a MIXED region
  // (MIXED) ---- M --> (MIXED)

  bitAttr_t RA = ATTR_NONE;
  unsigned StartBit = 0;

  std::vector<std::unique_ptr<Filter>> Filters;

  auto AddCandidateFilter = [&](unsigned StartBit, unsigned EndBit) {
    Filters.push_back(std::make_unique<Filter>(Encodings, EncodingIDs, StartBit,
                                               EndBit - StartBit));
  };

  unsigned FilterWidth = FilterBits.getBitWidth();
  for (unsigned BitIndex = 0; BitIndex != FilterWidth; ++BitIndex) {
    bitAttr_t bitAttr = BitAttrs[BitIndex];

    assert(bitAttr != ATTR_NONE && "Bit without attributes");

    switch (RA) {
    case ATTR_NONE:
      switch (bitAttr) {
      case ATTR_FILTERED:
        break;
      case ATTR_ALL_SET:
        StartBit = BitIndex;
        RA = ATTR_ALL_SET;
        break;
      case ATTR_ALL_UNSET:
        break;
      case ATTR_MIXED:
        StartBit = BitIndex;
        RA = ATTR_MIXED;
        break;
      default:
        llvm_unreachable("Unexpected bitAttr!");
      }
      break;
    case ATTR_ALL_SET:
      if (!AllowMixed && bitAttr != ATTR_ALL_SET)
        AddCandidateFilter(StartBit, BitIndex);
      switch (bitAttr) {
      case ATTR_FILTERED:
        RA = ATTR_NONE;
        break;
      case ATTR_ALL_SET:
        break;
      case ATTR_ALL_UNSET:
        RA = ATTR_NONE;
        break;
      case ATTR_MIXED:
        StartBit = BitIndex;
        RA = ATTR_MIXED;
        break;
      default:
        llvm_unreachable("Unexpected bitAttr!");
      }
      break;
    case ATTR_MIXED:
      if (AllowMixed && bitAttr != ATTR_MIXED)
        AddCandidateFilter(StartBit, BitIndex);
      switch (bitAttr) {
      case ATTR_FILTERED:
        StartBit = BitIndex;
        RA = ATTR_NONE;
        break;
      case ATTR_ALL_SET:
        StartBit = BitIndex;
        RA = ATTR_ALL_SET;
        break;
      case ATTR_ALL_UNSET:
        RA = ATTR_NONE;
        break;
      case ATTR_MIXED:
        break;
      default:
        llvm_unreachable("Unexpected bitAttr!");
      }
      break;
    case ATTR_ALL_UNSET:
      llvm_unreachable("regionAttr state machine has no ATTR_UNSET state");
    case ATTR_FILTERED:
      llvm_unreachable("regionAttr state machine has no ATTR_FILTERED state");
    }
  }

  // At the end, if we're still in ALL_SET or MIXED states, report a region
  switch (RA) {
  case ATTR_NONE:
    break;
  case ATTR_FILTERED:
    break;
  case ATTR_ALL_SET:
    if (!AllowMixed)
      AddCandidateFilter(StartBit, FilterWidth);
    break;
  case ATTR_ALL_UNSET:
    break;
  case ATTR_MIXED:
    if (AllowMixed)
      AddCandidateFilter(StartBit, FilterWidth);
    break;
  }

  // We have finished with the filter processings.  Now it's time to choose
  // the best performing filter.
  auto MaxIt = llvm::max_element(Filters, [](const std::unique_ptr<Filter> &A,
                                             const std::unique_ptr<Filter> &B) {
    return A->usefulness() < B->usefulness();
  });
  if (MaxIt == Filters.end() || (*MaxIt)->usefulness() == 0)
    return nullptr;
  return std::move(*MaxIt);
}

std::unique_ptr<Filter> FilterChooser::findBestFilter() const {
  // We maintain BIT_WIDTH copies of the bitAttrs automaton.
  // The automaton consumes the corresponding bit from each
  // instruction.
  //
  //   Input symbols: 0, 1, _ (unset), and . (any of the above).
  //   States:        NONE, FILTERED, ALL_SET, ALL_UNSET, and MIXED.
  //   Initial state: NONE.
  //
  // (NONE) ------- [01] -> (ALL_SET)
  // (NONE) ------- _ ----> (ALL_UNSET)
  // (ALL_SET) ---- [01] -> (ALL_SET)
  // (ALL_SET) ---- _ ----> (MIXED)
  // (ALL_UNSET) -- [01] -> (MIXED)
  // (ALL_UNSET) -- _ ----> (ALL_UNSET)
  // (MIXED) ------ . ----> (MIXED)
  // (FILTERED)---- . ----> (FILTERED)

  unsigned FilterWidth = FilterBits.getBitWidth();
  SmallVector<bitAttr_t, 128> BitAttrs(FilterWidth, ATTR_NONE);

  // FILTERED bit positions provide no entropy and are not worthy of pursuing.
  // Filter::recurse() set either 1 or 0 for each position.
  for (unsigned BitIndex = 0; BitIndex != FilterWidth; ++BitIndex)
    if (isPositionFiltered(BitIndex))
      BitAttrs[BitIndex] = ATTR_FILTERED;

  for (unsigned EncodingID : EncodingIDs) {
    const InstructionEncoding &Encoding = Encodings[EncodingID];
    KnownBits EncodingBits = Encoding.getMandatoryBits();

    for (unsigned BitIndex = 0; BitIndex != FilterWidth; ++BitIndex) {
      bool IsKnown = EncodingBits.Zero[BitIndex] || EncodingBits.One[BitIndex];
      switch (BitAttrs[BitIndex]) {
      case ATTR_NONE:
        if (IsKnown)
          BitAttrs[BitIndex] = ATTR_ALL_SET;
        else
          BitAttrs[BitIndex] = ATTR_ALL_UNSET;
        break;
      case ATTR_ALL_SET:
        if (!IsKnown)
          BitAttrs[BitIndex] = ATTR_MIXED;
        break;
      case ATTR_ALL_UNSET:
        if (IsKnown)
          BitAttrs[BitIndex] = ATTR_MIXED;
        break;
      case ATTR_MIXED:
      case ATTR_FILTERED:
        break;
      }
    }
  }

  // Try regions of consecutive known bit values first.
  if (std::unique_ptr<Filter> F =
          findBestFilter(BitAttrs, /*AllowMixed=*/false))
    return F;

  // Then regions of mixed bits (both known and unitialized bit values allowed).
  if (std::unique_ptr<Filter> F = findBestFilter(BitAttrs, /*AllowMixed=*/true))
    return F;

  // Heuristics to cope with conflict set {t2CMPrs, t2SUBSrr, t2SUBSrs} where
  // no single instruction for the maximum ATTR_MIXED region Inst{14-4} has a
  // well-known encoding pattern.  In such case, we backtrack and scan for the
  // the very first consecutive ATTR_ALL_SET region and assign a filter to it.
  if (EncodingIDs.size() == 3) {
    if (std::unique_ptr<Filter> F =
            findBestFilter(BitAttrs, /*AllowMixed=*/true, /*Greedy=*/false))
      return F;
  }

  // There is a conflict we could not resolve.
  return nullptr;
}

// Decides on the best configuration of filter(s) to use in order to decode
// the instructions.  A conflict of instructions may occur, in which case we
// dump the conflict set to the standard error.
void FilterChooser::doFilter() {
  assert(!EncodingIDs.empty() && "FilterChooser created with no instructions");

  // No filter needed.
  if (EncodingIDs.size() == 1) {
    SingletonEncodingID = EncodingIDs.front();
    return;
  }

  std::unique_ptr<Filter> BestFilter = findBestFilter();
  if (BestFilter) {
    applyFilter(*BestFilter);
    return;
  }

  // Print out useful conflict information for postmortem analysis.
  errs() << "Decoding Conflict:\n";
  dump();
  HasConflict = true;
}

void FilterChooser::dump() const {
  indent Indent(4);
  // Helps to keep the output right-justified.
  unsigned PadToWidth = getMaxEncodingWidth();

  // Dump filter stack.
  dumpStack(errs(), Indent, PadToWidth);

  // Dump encodings.
  for (unsigned EncodingID : EncodingIDs) {
    const InstructionEncoding &Encoding = Encodings[EncodingID];
    errs() << Indent << indent(PadToWidth - Encoding.getBitWidth());
    printKnownBits(errs(), Encoding.getMandatoryBits(), '_');
    errs() << "  " << Encoding.getName() << '\n';
  }
}

namespace {

class DecoderTreeBuilder {
  const CodeGenTarget &Target;
  ArrayRef<InstructionEncoding> Encodings;

public:
  DecoderTreeBuilder(const CodeGenTarget &Target,
                     ArrayRef<InstructionEncoding> Encodings)
      : Target(Target), Encodings(Encodings) {}

  std::unique_ptr<DecoderTreeNode> buildTree(const FilterChooser &FC) {
    return buildCheckAnyNode(FC);
  }

private:
  std::unique_ptr<DecoderTreeNode>
  buildTerminalNode(unsigned EncodingID, const KnownBits &FilterBits);

  std::unique_ptr<DecoderTreeNode> buildCheckAllOrSwitchNode(
      unsigned StartBit, unsigned NumBits,
      const std::map<uint64_t, std::unique_ptr<const FilterChooser>> &FCMap);

  std::unique_ptr<DecoderTreeNode> buildCheckAnyNode(const FilterChooser &FC);
};

} // namespace

static bool doesOpcodeNeedPredicate(const InstructionEncoding &Encoding) {
  std::vector<const Record *> Predicates =
      Encoding.getRecord()->getValueAsListOfDefs("Predicates");
  auto MCPredicates = make_filter_range(Predicates, [](const Record *R) {
    return R->getValue("AssemblerMatcherPredicate");
  });
  return !MCPredicates.empty();
}

static std::string getPredicateString(const InstructionEncoding &Encoding,
                                      StringRef TargetName) {
  std::vector<const Record *> Predicates =
      Encoding.getRecord()->getValueAsListOfDefs("Predicates");
  auto It = llvm::find_if(Predicates, [](const Record *R) {
    return R->getValueAsBit("AssemblerMatcherPredicate");
  });
  if (It == Predicates.end())
    return std::string();

  std::string PredicateString;
  raw_string_ostream OS(PredicateString);
  SubtargetFeatureInfo::emitMCPredicateCheck(OS, TargetName, Predicates);
  return PredicateString;
}

static void emitBinaryParser(raw_ostream &OS, indent Indent,
                             const InstructionEncoding &Encoding,
                             const OperandInfo &OpInfo) {
  if (OpInfo.HasNoEncoding) {
    // If an operand has no encoding, the old behavior is to not decode it
    // automatically and let the target do it. This is error-prone, so the
    // new behavior is to report an error.
    if (!IgnoreNonDecodableOperands)
      PrintError(Encoding.getRecord()->getLoc(),
                 "could not find field for operand '" + OpInfo.Name + "'");
    return;
  }
  // Special case for 'bits<0>'.
  if (OpInfo.Fields.empty() && !OpInfo.InitValue) {
    if (IgnoreNonDecodableOperands)
      return;
    assert(!OpInfo.Decoder.empty());
    // The operand has no encoding, so the corresponding argument is omitted.
    // This avoids confusion and allows the function to be overloaded if the
    // operand does have an encoding in other instructions.
    OS << Indent << "if (!Check(S, " << OpInfo.Decoder << "(MI, Decoder)))\n"
       << Indent << "  return MCDisassembler::Fail;\n";
    return;
  }

  if (OpInfo.fields().empty()) {
    // Only a constant part. The old behavior is to not decode this operand.
    if (IgnoreFullyDefinedOperands)
      return;
    // Initialize `tmp` with the constant part.
    OS << Indent << "tmp = " << format_hex(*OpInfo.InitValue, 0) << ";\n";
  } else if (OpInfo.fields().size() == 1 && !OpInfo.InitValue.value_or(0)) {
    // One variable part and no/zero constant part. Initialize `tmp` with the
    // variable part.
    auto [Base, Width, Offset] = OpInfo.fields().front();
    OS << Indent << "tmp = fieldFromInstruction(insn, " << Base << ", " << Width
       << ')';
    if (Offset)
      OS << " << " << Offset;
    OS << ";\n";
  } else {
    // General case. Initialize `tmp` with the constant part, if any, and
    // insert the variable parts into it.
    OS << Indent << "tmp = " << format_hex(OpInfo.InitValue.value_or(0), 0)
       << ";\n";
    for (auto [Base, Width, Offset] : OpInfo.fields())
      OS << Indent << "insertBits(tmp, fieldFromInstruction(insn, " << Base
         << ", " << Width << "), " << Offset << ", " << Width << ");\n";
  }

  StringRef Decoder = OpInfo.Decoder;
  if (!Decoder.empty()) {
    OS << Indent << "if (!Check(S, " << Decoder
       << "(MI, tmp, Address, Decoder))) { "
       << (OpInfo.HasCompleteDecoder ? "" : "DecodeComplete = false; ")
       << "return MCDisassembler::Fail; }\n";
  } else {
    OS << Indent << "MI.addOperand(MCOperand::createImm(tmp));\n";
  }
}

static void emitDecoder(raw_ostream &OS, indent Indent,
                        const InstructionEncoding &Encoding) {
  // If a custom instruction decoder was specified, use that.
  StringRef DecoderMethod = Encoding.getDecoderMethod();
  if (!DecoderMethod.empty()) {
    OS << Indent << "if (!Check(S, " << DecoderMethod
       << "(MI, insn, Address, Decoder))) { "
       << (Encoding.hasCompleteDecoder() ? "" : "DecodeComplete = false; ")
       << "return MCDisassembler::Fail; }\n";
    return;
  }

  for (const OperandInfo &Op : Encoding.getOperands())
    emitBinaryParser(OS, Indent, Encoding, Op);
}

static std::string getDecoderString(const InstructionEncoding &Encoding) {
  std::string DecoderString;
  raw_string_ostream S(DecoderString);
  indent Indent(UseFnTableInDecodeToMCInst ? 2 : 4);
  emitDecoder(S, Indent, Encoding);
  return DecoderString;
}

std::unique_ptr<DecoderTreeNode>
DecoderTreeBuilder::buildTerminalNode(unsigned EncodingID,
                                      const KnownBits &FilterBits) {
  const InstructionEncoding &Encoding = Encodings[EncodingID];
  auto N = std::make_unique<CheckAllNode>();

  if (doesOpcodeNeedPredicate(Encoding)) {
    std::string Predicate = getPredicateString(Encoding, Target.getName());
    N->addChild(std::make_unique<CheckPredicateNode>(std::move(Predicate)));
  }

  std::vector<FilterChooser::Island> Islands =
      FilterChooser::getIslands(Encoding.getMandatoryBits(), FilterBits);
  for (const FilterChooser::Island &Ilnd : reverse(Islands)) {
    N->addChild(std::make_unique<CheckFieldNode>(Ilnd.StartBit, Ilnd.NumBits,
                                                 Ilnd.FieldVal));
  }

  const KnownBits &InstBits = Encoding.getInstBits();
  const APInt &SoftFailMask = Encoding.getSoftFailMask();
  if (!SoftFailMask.isZero()) {
    APInt PositiveMask = InstBits.Zero & SoftFailMask;
    APInt NegativeMask = InstBits.One & SoftFailMask;
    N->addChild(std::make_unique<SoftFailNode>(PositiveMask.getZExtValue(),
                                               NegativeMask.getZExtValue()));
  }

  std::string DecoderIndex = getDecoderString(Encoding);
  N->addChild(std::make_unique<DecodeNode>(Encoding, DecoderIndex));

  return N;
}

std::unique_ptr<DecoderTreeNode> DecoderTreeBuilder::buildCheckAllOrSwitchNode(
    unsigned StartBit, unsigned NumBits,
    const std::map<uint64_t, std::unique_ptr<const FilterChooser>> &FCMap) {
  if (FCMap.size() == 1) {
    const auto &[FieldVal, ChildFC] = *FCMap.begin();
    auto N = std::make_unique<CheckAllNode>();
    N->addChild(std::make_unique<CheckFieldNode>(StartBit, NumBits, FieldVal));
    N->addChild(buildCheckAnyNode(*ChildFC));
    return N;
  }
  auto N = std::make_unique<SwitchFieldNode>(StartBit, NumBits);
  for (const auto &[FieldVal, ChildFC] : FCMap)
    N->addCase(FieldVal, buildCheckAnyNode(*ChildFC));
  return N;
}

std::unique_ptr<DecoderTreeNode>
DecoderTreeBuilder::buildCheckAnyNode(const FilterChooser &FC) {
  auto N = std::make_unique<CheckAnyNode>();
  if (FC.SingletonEncodingID) {
    N->addChild(buildTerminalNode(*FC.SingletonEncodingID, FC.FilterBits));
  } else {
    N->addChild(buildCheckAllOrSwitchNode(FC.StartBit, FC.NumBits,
                                          FC.FilterChooserMap));
  }
  if (FC.VariableFC) {
    N->addChild(buildCheckAnyNode(*FC.VariableFC));
  }

  return N;
}

std::unique_ptr<DecoderTreeNode>
llvm::buildDecoderTree(const CodeGenTarget &Target,
                       ArrayRef<InstructionEncoding> Encodings,
                       ArrayRef<unsigned> EncodingIDs) {
  FilterChooser FC(Encodings, EncodingIDs);
  if (FC.hasConflict())
    return nullptr;
  DecoderTreeBuilder TreeBuilder(Target, Encodings);
  return TreeBuilder.buildTree(FC);
}
