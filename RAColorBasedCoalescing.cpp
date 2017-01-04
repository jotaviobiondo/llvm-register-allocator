//===-- RAColorBasedCoalescing.cpp - Color-based Coalescing Register Allocator ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the MIT License.
// See the LICENSE file for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the RAColorBasedCoalescing function pass, which provides
// a implementation of the Coloring-based coalescing register allocator.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "CodeGen/AllocationOrder.h"
#include "CodeGen/LiveDebugVariables.h"
#include "CodeGen/SplitKit.h"
#include "/usr/local/src/llvm-build/llvm/lib/CodeGen/RegAllocBase.h"
#include "/usr/local/src/llvm-build/llvm/lib/CodeGen/Spiller.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/PassAnalysisSupport.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <cstdlib>
#include <map>
#include <set>
#include <queue>
#include <list>

using namespace llvm;

#define DEBUG_TYPE "regalloc"

namespace llvm {
  FunctionPass *createColorBasedRegAlloc();
}

static RegisterRegAlloc colorBasedCoalescingRegAlloc("colorBased",
                                                     "color-based coalescing register allocator",
                                                     createColorBasedRegAlloc);

namespace {
  struct CompSpillWeight {
    bool operator()(LiveInterval *A, LiveInterval *B) const {
      return A->weight < B->weight;
    }
  };
}


namespace {

  std::map<unsigned, std::set<unsigned>> InterferenceGraph;
  std::map<unsigned, int> Degree;

  class RAColorBasedCoalescing : public MachineFunctionPass, public RegAllocBase {
    // context
    MachineFunction *MF;

    // state
    std::unique_ptr<Spiller> SpillerInstance;
    std::priority_queue<LiveInterval*, std::vector<LiveInterval*>,
                        CompSpillWeight> Queue;

    // Scratch space.  Allocated here to avoid repeated malloc calls in
    // selectOrSplit().
    BitVector UsableRegs;

    std::unique_ptr<SplitAnalysis> SA;
    std::unique_ptr<SplitEditor> SE;

    //SlotIndexes *Indexes;
    MachineBlockFrequencyInfo *MBFI;
    MachineDominatorTree *DomTree;
    MachineLoopInfo *Loops;
    //EdgeBundles *Bundles;
    //SpillPlacement *SpillPlacer;
    LiveDebugVariables *DebugVars;
    AliasAnalysis *AA;


    private:
      void splitLiveRanges(MachineFunction &mf);

      void buildInterferenceGraph(MachineFunction &mf);

      void addInterferenceEdge();

      unsigned tryBlockSplit(LiveInterval &VirtReg, AllocationOrder &Order, SmallVectorImpl<unsigned> &NewVRegs);

    public:
      RAColorBasedCoalescing();

      /// Return the pass name.
      StringRef getPassName() const override { return "Color-based Coalescing Register Allocator"; }

      /// RAColorBasedCoalescing analysis usage.
      void getAnalysisUsage(AnalysisUsage &AU) const override;

      void releaseMemory() override;

      Spiller &spiller() override { return *SpillerInstance; }

      void enqueue(LiveInterval *LI) override {
        Queue.push(LI);
      }

      LiveInterval *dequeue() override {
        if (Queue.empty())
          return nullptr;
        LiveInterval *LI = Queue.top();
        Queue.pop();
        return LI;
      }

      unsigned selectOrSplit(LiveInterval &VirtReg,
                             SmallVectorImpl<unsigned> &SplitVRegs) override;

      /// Perform register allocation.
      bool runOnMachineFunction(MachineFunction &mf) override;

      MachineFunctionProperties getRequiredProperties() const override {
        return MachineFunctionProperties().set(
            MachineFunctionProperties::Property::NoPHIs);
      }

      // Helper for spilling all live virtual registers currently unified under preg
      // that interfere with the most recently queried lvr.  Return true if spilling
      // was successful, and append any new spilled/split intervals to splitLVRs.
      bool spillInterferences(LiveInterval &VirtReg, unsigned PhysReg,
                              SmallVectorImpl<unsigned> &SplitVRegs);

      static char ID;
  };

  char RAColorBasedCoalescing::ID = 0;

} // end anonymous namespace

RAColorBasedCoalescing::RAColorBasedCoalescing(): MachineFunctionPass(ID) {
  initializeLiveDebugVariablesPass(*PassRegistry::getPassRegistry());
  initializeLiveIntervalsPass(*PassRegistry::getPassRegistry());
  initializeSlotIndexesPass(*PassRegistry::getPassRegistry());
  initializeRegisterCoalescerPass(*PassRegistry::getPassRegistry());
  initializeMachineSchedulerPass(*PassRegistry::getPassRegistry());
  initializeLiveStacksPass(*PassRegistry::getPassRegistry());
  initializeMachineDominatorTreePass(*PassRegistry::getPassRegistry());
  initializeMachineLoopInfoPass(*PassRegistry::getPassRegistry());
  initializeVirtRegMapPass(*PassRegistry::getPassRegistry());
  initializeLiveRegMatrixPass(*PassRegistry::getPassRegistry());
}

void RAColorBasedCoalescing::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<LiveIntervals>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<LiveDebugVariables>();
  AU.addPreserved<LiveDebugVariables>();
  AU.addRequired<LiveStacks>();
  AU.addPreserved<LiveStacks>();
  AU.addRequired<MachineBlockFrequencyInfo>();
  AU.addPreserved<MachineBlockFrequencyInfo>();
  AU.addRequired<MachineDominatorTree>();
  AU.addPreserved<MachineDominatorTree>();
  AU.addRequired<MachineLoopInfo>();
  AU.addPreserved<MachineLoopInfo>();
  AU.addRequired<VirtRegMap>();
  AU.addPreserved<VirtRegMap>();
  AU.addRequired<LiveRegMatrix>();
  AU.addPreserved<LiveRegMatrix>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void RAColorBasedCoalescing::releaseMemory() {
  SpillerInstance.reset();
}


// Spill or split all live virtual registers currently unified under PhysReg
// that interfere with VirtReg. The newly spilled or split live intervals are
// returned by appending them to SplitVRegs.
bool RAColorBasedCoalescing::spillInterferences(LiveInterval &VirtReg, unsigned PhysReg, SmallVectorImpl<unsigned> &SplitVRegs) {
  // Record each interference and determine if all are spillable before mutating
  // either the union or live intervals.
  SmallVector<LiveInterval*, 8> Intfs;

  // Collect interferences assigned to any alias of the physical register.
  for (MCRegUnitIterator Units(PhysReg, TRI); Units.isValid(); ++Units) {
    LiveIntervalUnion::Query &Q = Matrix->query(VirtReg, *Units);
    Q.collectInterferingVRegs();
    if (Q.seenUnspillableVReg())
      return false;
    for (unsigned i = Q.interferingVRegs().size(); i; --i) {
      LiveInterval *Intf = Q.interferingVRegs()[i - 1];
      if (!Intf->isSpillable() || Intf->weight > VirtReg.weight)
        return false;
      Intfs.push_back(Intf);
    }
  }
  DEBUG(dbgs() << "spilling " << TRI->getName(PhysReg) <<
        " interferences with " << VirtReg << "\n");
  assert(!Intfs.empty() && "expected interference");

  // Spill each interfering vreg allocated to PhysReg or an alias.
  for (unsigned i = 0, e = Intfs.size(); i != e; ++i) {
    LiveInterval &Spill = *Intfs[i];

    // Skip duplicates.
    if (!VRM->hasPhys(Spill.reg))
      continue;

    // Deallocate the interfering vreg by removing it from the union.
    // A LiveInterval instance may not be in a union during modification!
    Matrix->unassign(Spill);

    // Spill the extracted interval.
    LiveRangeEdit LRE(&Spill, SplitVRegs, *MF, *LIS, VRM, nullptr, &DeadRemats);
    spiller().spill(LRE);
  }
  return true;
}

// Driver for the register assignment and splitting heuristics.
// Manages iteration over the LiveIntervalUnions.
//
// This is a minimal implementation of register assignment and splitting that
// spills whenever we run out of registers.
//
// selectOrSplit can only be called once per live virtual register. We then do a
// single interference test for each register the correct class until we find an
// available register. So, the number of interference tests in the worst case is
// |vregs| * |machineregs|. And since the number of interference tests is
// minimal, there is no value in caching them outside the scope of
// selectOrSplit().
unsigned RAColorBasedCoalescing::selectOrSplit(LiveInterval &VirtReg, SmallVectorImpl<unsigned> &SplitVRegs) {
  // Populate a list of physical register spill candidates.
  SmallVector<unsigned, 8> PhysRegSpillCands;

  // Check for an available register in this class.
  AllocationOrder Order(VirtReg.reg, *VRM, RegClassInfo, Matrix);
  while (unsigned PhysReg = Order.next()) {
    // Check for interference in PhysReg
    switch (Matrix->checkInterference(VirtReg, PhysReg)) {
    case LiveRegMatrix::IK_Free:
      // PhysReg is available, allocate it.
      return PhysReg;

    case LiveRegMatrix::IK_VirtReg:
      // Only virtual registers in the way, we may be able to spill them.
      PhysRegSpillCands.push_back(PhysReg);
      continue;

    default:
      // RegMask or RegUnit interference.
      continue;
    }
  }

  // Try to spill another interfering reg with less spill weight.
  for (SmallVectorImpl<unsigned>::iterator PhysRegI = PhysRegSpillCands.begin(),
       PhysRegE = PhysRegSpillCands.end(); PhysRegI != PhysRegE; ++PhysRegI) {
    if (!spillInterferences(VirtReg, *PhysRegI, SplitVRegs))
      continue;

    assert(!Matrix->checkInterference(VirtReg, *PhysRegI) &&
           "Interference after spill.");
    // Tell the caller to allocate to this newly freed physical register.
    return *PhysRegI;
  }

  // No other spill candidates were found, so spill the current VirtReg.
  DEBUG(dbgs() << "spilling: " << VirtReg << '\n');
  if (!VirtReg.isSpillable())
    return ~0u;
  LiveRangeEdit LRE(&VirtReg, SplitVRegs, *MF, *LIS, VRM, nullptr, &DeadRemats);
  spiller().spill(LRE);

  // The live virtual register requesting allocation was spilled, so tell
  // the caller not to allocate anything during this round.
  return 0;
}

//===---------------------------------===//
//   Coloring-Based Coalescing Methods
//===---------------------------------===//

void RAColorBasedCoalescing::splitLiveRanges(MachineFunction &mf) {

}

// Add a interference edge on the Interference Graph
void RAColorBasedCoalescing::addInterferenceEdge() {

}

// Builds the Interference Graph
/*void RAColorBasedCoalescing::buildInterferenceGraph(MachineFunction &mf) {
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    // reg ID
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    dbgs() << "index2VirtReg i: " << i << " Reg: " << Reg << "\n";
    dbgs() << PrintReg(Reg, TRI) << "\n";
    // if is not a DEBUG register
    if (MRI->reg_nodbg_empty(Reg)) {
      dbgs() << "It is a DEBUG register.\n";
      continue;
    }
    // get the respective LiveInterval
    LiveInterval *VirtReg = &LIS->getInterval(Reg);
    dbgs() << "LiveInterval: " << VirtReg << "\n";
  }
}*/

// void RAColorBasedCoalescing::printInterferenceGraph() {
//
// }
//
// bool RAColorBasedCoalescing::isExtendedColor(int color) {
//   return color < 0;
// }
//
// void RAColorBasedCoalescing::assignExtendedColor() {
//
// }

bool RAColorBasedCoalescing::runOnMachineFunction(MachineFunction &mf) {
  dbgs() << "\n********** COLORING-BASED COALESCING REGISTER ALLOCATION **********\n"
               << "********** Function: "
               << mf.getName() << '\n';

  MF = &mf;
  RegAllocBase::init(getAnalysis<VirtRegMap>(),
                     getAnalysis<LiveIntervals>(),
                     getAnalysis<LiveRegMatrix>());

  //Indexes = &getAnalysis<SlotIndexes>();
  MBFI = &getAnalysis<MachineBlockFrequencyInfo>();
  DomTree = &getAnalysis<MachineDominatorTree>();



  calculateSpillWeightsAndHints(*LIS, *MF, VRM,
                                getAnalysis<MachineLoopInfo>(),
                                getAnalysis<MachineBlockFrequencyInfo>());

  SpillerInstance.reset(createInlineSpiller(*this, *MF, *VRM));

  Loops = &getAnalysis<MachineLoopInfo>();
  //Bundles = &getAnalysis<EdgeBundles>();
  //SpillPlacer = &getAnalysis<SpillPlacement>();
  DebugVars = &getAnalysis<LiveDebugVariables>();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();


  dbgs() << "********** Number of virtual registers: " << MRI->getNumVirtRegs() << "\n\n";

  SA.reset(new SplitAnalysis(*VRM, *LIS, *Loops));
  SE.reset(new SplitEditor(*SA, *AA, *LIS, *VRM, *DomTree, *MBFI));
  //buildInterferenceGraph(mf);

  /*for (TargetRegisterInfo::regclass_iterator RCi = TRI->regclass_begin(), RCe = TRI->regclass_end(); RCi != RCe; ++RCi) {
    dbgs() << "RCI: " << RCi << " - " << (*RCi)->getNumRegs() << "\n";
  }

  for (MachineFunction::iterator MBBi = mf.begin(), MBBe = mf.end(); MBBi != MBBe; ++MBBi) {
    dbgs() << "Building interference graph on " << *MBBi << "\n";
  }*/

  typedef SmallVector<unsigned, 4> VirtRegVec;

  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    // reg ID
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    // if is not a DEBUG register
    if (MRI->reg_nodbg_empty(Reg))
      continue;

    // get the respective LiveInterval
    LiveInterval *VirtReg = &LIS->getInterval(Reg);
    //dbgs() << *VirtReg << '\n';
    VirtRegVec SplitVRegs;
    AllocationOrder Order(VirtReg->reg, *VRM, RegClassInfo, Matrix);
    tryBlockSplit(*VirtReg, Order, SplitVRegs);

    dbgs() << "\n------------------------------------\n";
    for (VirtRegVec::iterator I = SplitVRegs.begin(), E = SplitVRegs.end(); I != E; ++I) {
      LiveInterval *SplitVirtReg = &LIS->getInterval(*I);
      dbgs() << *SplitVirtReg << '\n';
    }
    dbgs() << "------------------------------------\n";
  }

  dbgs() << "\n2222********** Number of virtual registers: " << MRI->getNumVirtRegs() << "\n\n";
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    // reg ID
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    // if is not a DEBUG register
    if (MRI->reg_nodbg_empty(Reg))
      continue;

    // get the respective LiveInterval
    LiveInterval *VirtReg = &LIS->getInterval(Reg);
    dbgs() << *VirtReg << '\n';
  }

  /*bool change = true;
  //split
  while (change) {
    //change = buildInterferenceGraph
    while (totalSpillCost < totalPreviousSpillCost && i < MAX_ITERATIONS) {
      //totalPreviousSpillCost = totalSpillCost
      //simplify (remove from interference graph, add to stack)
      //coloring (with no limit of colors)
      //coalescing
      //totalSpillCost = ...
      //clear
    }
    //simplify
    //coloring
    // if(no spill) break
  }*/


  allocatePhysRegs();
  postOptimization();

  // Diagnostic output before rewriting
  dbgs() << "\nPost alloc VirtRegMap:\n" << *VRM << "\n";

  releaseMemory();
  return true;
}

unsigned RAColorBasedCoalescing::tryBlockSplit(LiveInterval &VirtReg, AllocationOrder &Order,
                                 SmallVectorImpl<unsigned> &NewVRegs) {
  SA->analyze(&VirtReg);
  assert(&SA->getParent() == &VirtReg && "Live range wasn't analyzed");
  unsigned Reg = VirtReg.reg;
  bool SingleInstrs = RegClassInfo.isProperSubClass(MRI->getRegClass(Reg));
  LiveRangeEdit LREdit(&VirtReg, NewVRegs, *MF, *LIS, VRM, nullptr);
  SE->reset(LREdit);
  ArrayRef<SplitAnalysis::BlockInfo> UseBlocks = SA->getUseBlocks();

  for (unsigned i = 0; i != UseBlocks.size(); ++i) {
    const SplitAnalysis::BlockInfo &BI = UseBlocks[i];
    if (SA->shouldSplitSingleBlock(BI, SingleInstrs)) {
      dbgs() << "Splitando!!!!!!!!!";
      SE->splitSingleBlock(BI);
    }
  }
  // No blocks were split.
  if (LREdit.empty())
    return 0;

  // We did split for some blocks.
  SmallVector<unsigned, 8> IntvMap;
  SE->finish(&IntvMap);

  // Tell LiveDebugVariables about the new ranges.
  DebugVars->splitRegister(Reg, LREdit.regs(), *LIS);

  return 0;
}

FunctionPass *llvm::createColorBasedRegAlloc() {
  return new RAColorBasedCoalescing();
}
