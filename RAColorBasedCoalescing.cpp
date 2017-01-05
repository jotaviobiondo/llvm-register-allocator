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
#include "/usr/local/src/llvm-build/llvm/lib/CodeGen/RegAllocBase.h"
#include "/usr/local/src/llvm-build/llvm/lib/CodeGen/Spiller.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
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
  std::map<unsigned, bool> OnStack;
  std::queue<unsigned> ColoringStack;
  std::set<unsigned> Colored;
  BitVector Allocatable;
  std::set<unsigned> PhysicalRegisters;

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

    private:
      void algorithm(MachineFunction &mf);

      void split(MachineFunction &mf);

      void renumber(MachineFunction &mf);

      void buildInterferenceGraph(MachineFunction &mf);

      void addInterferenceEdge();

      void calculateSpillCosts(MachineFunction &mf);

      void simplify(MachineFunction &mf);

      void biased_select_extend(MachineFunction &mf);

      void coalescing(MachineFunction &mf);

      bool save_confirm(MachineFunction &mf);

      void clear(MachineFunction &mf);

      void biased_select(MachineFunction &mf);

      bool confirm(MachineFunction &mf);

      void spillCode(MachineFunction &mf);

      bool overlapsFrom1(const LiveRange *VR, const LiveRange *other) const;

      void printGraph();

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
  AU.addRequiredID(MachineDominatorsID);
  AU.addPreservedID(MachineDominatorsID);
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
void RAColorBasedCoalescing::algorithm(MachineFunction &mf){
  bool spill = true;
  bool improvement = true;

  split(mf);
  while (spill){
    renumber(mf);
    buildInterferenceGraph(mf);
    calculateSpillCosts(mf);

    while(improvement){
      //BACKUP

      simplify(mf);
      biased_select_extend(mf);
      coalescing(mf);

      if(save_confirm(mf)){
        //BACKUP = ATUAL
        //COSTS_ANTERIOR = COSTS_ATUAL 
        improvement = true;
      } else {
        //ATUAL = BACKUP
        improvement = false;
      }
      
      clear(mf);
    }

    simplify(mf);
    biased_select(mf);
    if(confirm(mf)){
      spill = true;
      spillCode(mf);
    } else {
      spill = false;
    }
  }
}


void RAColorBasedCoalescing::split(MachineFunction &mf) {

}

void RAColorBasedCoalescing::renumber(MachineFunction &mf){

}

// Add a interference edge on the Interference Graph
void RAColorBasedCoalescing::addInterferenceEdge() {
  
}

// Builds the Interference Graph
void RAColorBasedCoalescing::buildInterferenceGraph(MachineFunction &mf) {
  //Declaracao só para n ter q subir toda hora pra lembrar
  //std::map<unsigned, std::set<unsigned>> InterferenceGraph;
  //std::map<unsigned, int> Degree;
  //std::map<unsigned, bool> OnStack;
  //std::set<unsigned> Colored;
  //std::queue<unsigned> ColoringStack;
  //BitVector Allocatable;
  //std::set<unsigned> PhysicalRegisters;
  
  int num = 0;
  for(unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i){

    //reg ID
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    if(MRI->reg_nodbg_empty(Reg)) {
      //dbgs() << "DEBUG Register\n\n";
      continue;
    }
    num++;

    //get the respective LiveInterval
    LiveInterval *VirtReg = &LIS->getInterval(Reg);
    unsigned vReg = VirtReg->reg;
    
    OnStack[vReg] = false;
    
    //VER PQ ELE INSERE ESSE 0
    //InterferenceGraph[vReg].insert(0);

    for(unsigned j = 0, r = MRI->getNumVirtRegs(); j != r; ++j){
      unsigned Reg1 = TargetRegisterInfo::index2VirtReg(j);
      if(MRI->reg_nodbg_empty(Reg1)){
          continue;
      }
      LiveInterval *VirtReg1 = &LIS->getInterval(Reg1);
      unsigned vReg1 = VirtReg1->reg;

      if(VirtReg == VirtReg1){
        continue;
      }

      if(VirtReg->overlaps(*VirtReg1)){
        if(!InterferenceGraph[vReg].count(vReg1)){
          InterferenceGraph[vReg].insert(vReg1);
          Degree[vReg]++;
        }
        if(!InterferenceGraph[vReg1].count(vReg)){
          InterferenceGraph[vReg1].insert(vReg);
          Degree[vReg1]++;
        }
      } 
    }
  }
  //printGraph();
  errs( ) << "\nVirtual registers: " << num << "\n";
}

void RAColorBasedCoalescing::printGraph(){
  for(std::map<unsigned, std::set<unsigned>> :: iterator j = InterferenceGraph.begin(); j != InterferenceGraph.end(); j++){
    dbgs() << "Numero de Interferencias " << j->first << " => " << Degree[j->first] << "\n"; 
    std::set<unsigned> lista = j->second;
    dbgs() << "Registradores com interferencia:\n";
    for(std::set<unsigned> :: iterator k = j->second.begin(); k != j->second.end(); k++){
      dbgs() << *k << "\n";
    }
    dbgs() << "\n";
  }
}

bool RAColorBasedCoalescing::overlapsFrom1(const LiveRange *VR, const LiveRange *other) const {
  assert(!empty() && "empty range");
  llvm::LiveRange::const_iterator StartPos = other->begin();
  llvm::LiveRange::const_iterator i = VR->begin();
  llvm::LiveRange::const_iterator ie = VR->end();
  llvm::LiveRange::const_iterator j = StartPos;
  llvm::LiveRange::const_iterator je = other->end();

  //dbgs() << "\ni => " << *i; 
  //dbgs() << "\nie => " << *ie; 
  //dbgs() << "\nj => " << *j; 
  //dbgs() << "\nje => " << *je << "\n\n";
   
  assert((StartPos->start <= i->start || StartPos == other.begin()) && StartPos != other.end() && "Bogus start position hint!");


  if (i->start < j->start) {
    i = std::upper_bound(i, ie, j->start);
    if (i != VR->begin()) --i;
  } else if (j->start < i->start) {
    ++StartPos;
    if (StartPos != other->end() && StartPos->start <= i->start) {
      assert(StartPos < other->end() && i < VR->end());
      j = std::upper_bound(j, je, i->start);
      if (j != other->begin()) --j;
    }
  } else {
    //dbgs() << "PRIMEIRO TRUE\n";
    return true;
  }
  
  if (j == je) {
    //dbgs() << "FALSE J == JE\n";
    return false;
  }
  
  while (i != ie) {
    if (i->start > j->start) {
      std::swap(i, j);
      std::swap(ie, je);
    }
  
    if (i->end > j->start){
      //dbgs() << "SEGUNDO TRUE \n";
      return true;
    }
  
    ++i;
  }
  
  //dbgs() << "CHEGA ATE AQUI!\n"; 
  return false;
}

void RAColorBasedCoalescing::calculateSpillCosts(MachineFunction &mf){

}

void RAColorBasedCoalescing::simplify(MachineFunction &mf){
  unsigned min = 0;
  for(std::map<unsigned, std::set<unsigned>> :: iterator i = InterferenceGraph.begin(); i != InterferenceGraph.end(); i++){
    if(!OnStack[i->first] && (min == 0 || Degree[i->first] < Degree[min])){
      min = i->first;
    }
  }

  //graph empty
  if(min == 0){
    return;
  }

  OnStack[min] = true;
  ColoringStack.push(min);
  //dbgs() <<"Enfileirou => " << ColoringStack.back() << "\n";

  for(std::map<unsigned, std::set<unsigned>> :: iterator j = InterferenceGraph.begin(); j != InterferenceGraph.end(); j++){
    if(j->second.count(min)){
      Degree[j->first]--;
    }
  }

  simplify(mf);

}

void RAColorBasedCoalescing::biased_select_extend(MachineFunction &mf){

}

void RAColorBasedCoalescing::coalescing(MachineFunction &mf){

}

bool RAColorBasedCoalescing::save_confirm(MachineFunction &mf){
  return false;
}

void RAColorBasedCoalescing::clear(MachineFunction &mf){

}

void RAColorBasedCoalescing::biased_select(MachineFunction &mf){

}

bool RAColorBasedCoalescing::confirm(MachineFunction &mf){
  return false;
}

void RAColorBasedCoalescing::spillCode(MachineFunction &mf){

}

bool RAColorBasedCoalescing::runOnMachineFunction(MachineFunction &mf) {
  dbgs() << "\n********** COLORING-BASED COALESCING REGISTER ALLOCATION **********\n"
               << "********** Function: "
               << mf.getName() << '\n';

  MF = &mf;
  RegAllocBase::init(getAnalysis<VirtRegMap>(),
                     getAnalysis<LiveIntervals>(),
                     getAnalysis<LiveRegMatrix>());

  calculateSpillWeightsAndHints(*LIS, *MF, VRM,
                                getAnalysis<MachineLoopInfo>(),
                                getAnalysis<MachineBlockFrequencyInfo>());

  SpillerInstance.reset(createInlineSpiller(*this, *MF, *VRM));


  dbgs() << "********** Number of virtual registers: " << MRI->getNumVirtRegs() << "\n\n";

  algorithm(mf);
  
  /*for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
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

FunctionPass *llvm::createColorBasedRegAlloc() {
  return new RAColorBasedCoalescing();
}
