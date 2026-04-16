from __future__ import annotations
from enum       import Enum
from dataclasses import dataclass, field
from typing     import Optional, List, Dict, Tuple
import textwrap, json

# Configuration
CACHE_LINES  = 4           # direct-mapped, 4 sets
BLOCK_WORDS  = 4           # words per block
ADDR_BITS    = 8
OFFSET_BITS  = 2           # log2(BLOCK_WORDS)
INDEX_BITS   = 2           # log2(CACHE_LINES)
TAG_BITS     = ADDR_BITS - INDEX_BITS - OFFSET_BITS   # 4 bits
MEM_LATENCY  = 3           # cycles to complete a memory operation

# Enumerations
class State(Enum):
    IDLE        = "IDLE"
    COMPARE_TAG = "COMPARE_TAG"
    ALLOCATE    = "ALLOCATE"
    WRITE_BACK  = "WRITE_BACK"

class ReqType(Enum):
    READ  = "READ"
    WRITE = "WRITE"

# Data Classes
@dataclass
class CPURequest:
    req_type : ReqType
    address  : int
    data     : Optional[int] = None

    def __str__(self):
        t = f"tag=0x{(self.address >> (OFFSET_BITS+INDEX_BITS)) & 0xF:X}"
        i = f"idx={(self.address >> OFFSET_BITS) & 0x3}"
        o = f"off={self.address & 0x3}"
        if self.req_type == ReqType.WRITE:
            return f"WRITE  addr=0x{self.address:02X} ({t},{i},{o})  data=0x{self.data:02X}"
        return     f"READ   addr=0x{self.address:02X} ({t},{i},{o})"

@dataclass
class CacheLine:
    valid : bool       = False
    dirty : bool       = False
    tag   : int        = 0
    data  : List[int]  = field(default_factory=lambda: [0]*BLOCK_WORDS)

    def snapshot(self) -> str:
        return (f"V={int(self.valid)} D={int(self.dirty)} "
                f"tag=0x{self.tag:X}  [{', '.join(f'0x{d:02X}' for d in self.data)}]")

@dataclass
class CycleLog:
    cycle       : int
    state_in    : State
    state_out   : State
    description : str
    signals     : Dict
    hit         : bool
    cache_lines : List[str]

# Memory
class Memory:
    def __init__(self, size: int = 256):
        self.mem         : List[int]         = [0] * size
        self._busy       : bool              = False
        self._countdown  : int               = 0
        self._pending    : Optional[tuple]   = None   # ('r'|'w', base, data)

    # setup helpers
    def preset(self, addr: int, value: int):
        self.mem[addr] = value

    def preset_block(self, base: int, values: List[int]):
        for i, v in enumerate(values):
            self.mem[base + i] = v

    # block operations
    def start_read(self, block_base: int) -> bool:
        if self._busy: return False
        self._busy = True;  self._countdown = MEM_LATENCY
        self._pending = ('r', block_base, None)
        return True

    def start_write(self, block_base: int, data: List[int]) -> bool:
        if self._busy: return False
        self._busy = True;  self._countdown = MEM_LATENCY
        self._pending = ('w', block_base, list(data))
        return True

    def tick(self) -> Tuple[bool, Optional[List[int]]]:
        if not self._busy: return False, None
        self._countdown -= 1
        if self._countdown > 0: return False, None
        op, base, wdata = self._pending
        self._busy = False
        if op == 'r':
            return True, list(self.mem[base: base + BLOCK_WORDS])
        else:
            for i, d in enumerate(wdata):
                self.mem[base + i] = d
            return True, None

# Address Decoder
def decode(addr: int) -> Tuple[int,int,int,int]:
    off   = addr & ((1 << OFFSET_BITS) - 1)
    idx   = (addr >> OFFSET_BITS)             & ((1 << INDEX_BITS) - 1)
    tag   = (addr >> (OFFSET_BITS+INDEX_BITS)) & ((1 << TAG_BITS)  - 1)
    base  = addr & ~((1 << OFFSET_BITS) - 1)
    return tag, idx, off, base

# Cache Controller FSM
class CacheController:
    def __init__(self, memory: Memory):
        self.memory     = memory
        self.cache      = [CacheLine() for _ in range(CACHE_LINES)]
        self.state      = State.IDLE
        self.cycle      = 0
        self.log        : List[CycleLog] = []

        # Current in-flight request
        self.cur_req    : Optional[CPURequest] = None
        self.cur_tag    = 0
        self.cur_idx    = 0
        self.cur_off    = 0
        self.cur_base   = 0

        # Output signals (visible to CPU / bus)
        self.cpu_stall   = False
        self.cpu_dout   : Optional[int] = None
        self.hit         = False
        self.mem_read    = False
        self.mem_write   = False

    # private helpers
    @property
    def _line(self) -> CacheLine:
        return self.cache[self.cur_idx]

    def _is_hit(self) -> bool:
        l = self._line
        return l.valid and l.tag == self.cur_tag

    def _record(self, prev: State, desc: str):
        self.log.append(CycleLog(
            cycle       = self.cycle,
            state_in    = prev,
            state_out   = self.state,
            description = desc,
            signals     = {
                "cpu_stall"  : self.cpu_stall,
                "hit"        : self.hit,
                "cpu_dout"   : f"0x{self.cpu_dout:02X}" if self.cpu_dout is not None else "-",
                "mem_read"   : self.mem_read,
                "mem_write"  : self.mem_write,
            },
            hit         = self.hit,
            cache_lines = [f"[{i}] {l.snapshot()}" for i,l in enumerate(self.cache)],
        ))

    # public interface
    def tick(self, cpu_req: Optional[CPURequest] = None):
        self.cycle += 1
        prev = self.state

        # Reset pulse signals
        self.hit = False
        self.cpu_dout   = None
        self.mem_read   = False
        self.mem_write  = False

        {
            State.IDLE        : self._idle,
            State.COMPARE_TAG : self._compare_tag,
            State.ALLOCATE    : self._allocate,
            State.WRITE_BACK  : self._write_back,
        }[self.state](cpu_req)

        self._record(prev, self._describe(prev))

    # state handlers
    def _idle(self, req):
        desc_note = ""
        if req is not None:
            self.cur_req = req
            self.cur_tag, self.cur_idx, self.cur_off, self.cur_base = decode(req.address)
            self.cpu_stall = True
            self.state     = State.COMPARE_TAG
            desc_note = f"received {req}"
        else:
            self.cpu_stall = False

    def _compare_tag(self, _=None):
        line = self._line
        if self._is_hit():
            self.hit = True
            if self.cur_req.req_type == ReqType.READ:
                self.cpu_dout  = line.data[self.cur_off]
                self.cpu_stall = False
                self.state     = State.IDLE
            else:                                   # write hit
                line.data[self.cur_off] = self.cur_req.data
                line.dirty = True
                self.cpu_stall = False
                self.state     = State.IDLE
        else:
            self.hit = False
            if line.valid and line.dirty:           # dirty eviction needed
                old_base = (line.tag << (OFFSET_BITS+INDEX_BITS)) | (self.cur_idx << OFFSET_BITS)
                self.memory.start_write(old_base, line.data)
                self.mem_write = True
                self.state     = State.WRITE_BACK
            else:                                   # clean / invalid → allocate
                self.memory.start_read(self.cur_base)
                self.mem_read = True
                self.state    = State.ALLOCATE

    def _allocate(self, _=None):
        self.mem_read = True
        done, block = self.memory.tick()
        if done:
            line       = self._line
            line.valid = True
            line.dirty = False
            line.tag   = self.cur_tag
            line.data  = list(block)
            if self.cur_req.req_type == ReqType.READ:
                self.cpu_dout = line.data[self.cur_off]
            else:
                line.data[self.cur_off] = self.cur_req.data
                line.dirty = True
            self.cpu_stall = False
            self.mem_read  = False
            self.state     = State.IDLE

    def _write_back(self, _=None):
        self.mem_write = True
        done, _ = self.memory.tick()
        if done:
            self.memory.start_read(self.cur_base)
            self.mem_write = False
            self.mem_read  = True
            self.state     = State.ALLOCATE

    def _describe(self, prev: State) -> str:
        s = self.state
        if prev == State.IDLE and s == State.COMPARE_TAG:
            return f"New request latched → tag compare"
        if prev == State.COMPARE_TAG and s == State.IDLE:
            return f"Cache HIT → data served, back to IDLE"
        if prev == State.COMPARE_TAG and s == State.ALLOCATE:
            return f"Cache MISS (clean/invalid) → fetch from memory"
        if prev == State.COMPARE_TAG and s == State.WRITE_BACK:
            return f"Cache MISS + dirty → write-back old block first"
        if prev == State.ALLOCATE and s == State.ALLOCATE:
            return f"Waiting for memory read (countdown={self.memory._countdown})"
        if prev == State.ALLOCATE and s == State.IDLE:
            return f"Block loaded, request served → back to IDLE"
        if prev == State.WRITE_BACK and s == State.WRITE_BACK:
            return f"Waiting for memory write (countdown={self.memory._countdown})"
        if prev == State.WRITE_BACK and s == State.ALLOCATE:
            return f"Write-back done → now fetching new block"
        if prev == State.IDLE and s == State.IDLE:
            return "Idle, no request"
        return f"{prev.value} → {s.value}"

# CPU
class CPU:
    def __init__(self, requests: List[CPURequest]):
        self._queue   = list(requests)
        self.waiting  = False
        self.current  : Optional[CPURequest] = None
        self.results  : List[Dict] = []

    def tick(self, stall: bool, dout: Optional[int]) -> Optional[CPURequest]:
        if self.waiting and not stall:
            self.results.append({"req": self.current, "data": dout})
            self.waiting = False
            self.current = None
        if not self.waiting and self._queue:
            self.current = self._queue.pop(0)
            self.waiting = True
            return self.current
        return None

    @property
    def done(self) -> bool:
        return not self.waiting and not self._queue

# Simulation Runner
def simulate(label: str,
             requests: List[CPURequest],
             mem_init: Dict[int,int] = None,
             max_cycles: int = 300) -> Tuple[List[CycleLog], List[Dict], List[CacheLine]]:

    mem  = Memory()
    if mem_init:
        for a, v in mem_init.items():
            mem.preset(a, v)

    ctrl = CacheController(mem)
    cpu  = CPU(requests)

    stall = False
    dout  = None

    for _ in range(max_cycles):
        req  = cpu.tick(stall, dout)
        ctrl.tick(req)
        stall = ctrl.cpu_stall
        dout  = ctrl.cpu_dout
        if cpu.done and ctrl.state == State.IDLE:
            break

    return ctrl.log, cpu.results, ctrl.cache


# Formatted Printer
def print_simulation(label: str, requests, mem_init=None):
    print(f"\n{'━'*72}")
    print(f"  ▶  {label}")
    print(f"{'━'*72}")
    print(f"  Requests:")
    for r in requests:
        print(f"    • {r}")
    print()

    logs, results, final_cache = simulate(label, requests, mem_init)

    # Print cycle table
    print(f"  {'Cyc':>3}  {'State In':<12} {'→':1} {'State Out':<12}  {'H':1}  {'STALL':5}  "
          f"{'MREAD':5}  {'MWRITE':6}  Description")
    print(f"  {'─'*78}")
    for e in logs:
        hit   = '✓' if e.hit else ' '
        s     = e.signals
        print(f"  {e.cycle:>3}  {e.state_in.value:<12} → {e.state_out.value:<12}  "
              f"{hit:1}  {str(s['cpu_stall']):5}  {str(s['mem_read']):5}  "
              f"{str(s['mem_write']):6}  {e.description}")

    print(f"\n  Results from CPU:")
    for r in results:
        req = r['req']
        if req.req_type == ReqType.READ:
            d = r['data']
            dstr = f"0x{d:02X}" if d is not None else "None"
            print(f"    ✓  {req}  →  read_data = {dstr}")
        else:
            print(f"    ✓  {req}  →  written")

    print(f"\n  Final Cache State:")
    for i, cl in enumerate(final_cache):
        flag = " [DIRTY]" if cl.dirty else ""
        print(f"    [{i}] {cl.snapshot()}{flag}")

    return logs, results, final_cache

# Main
if __name__ == "__main__":

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║      FSM-Based Direct-Mapped Cache Controller Simulation             ║")
    print("║      Write-Back | Allocate-on-Write | 4 Lines × 4 Words             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  Address: {ADDR_BITS}-bit  →  tag[{TAG_BITS}] | index[{INDEX_BITS}] | offset[{OFFSET_BITS}]")
    print(f"  Cache:   {CACHE_LINES} lines × {BLOCK_WORDS} words   Memory latency: {MEM_LATENCY} cycles")

    # Scenario 1: Cold Read Miss
    print_simulation(
        "Scenario 1 – Read Miss (Cold Start, empty cache)",
        requests=[CPURequest(ReqType.READ, 0x05)],
        mem_init={0x04:0xAA, 0x05:0xBB, 0x06:0xCC, 0x07:0xDD},
    )

    # Scenario 2: Read Miss → Subsequent Read Hits (spatial locality)
    print_simulation(
        "Scenario 2 – Read Miss then Read Hits (spatial locality)",
        requests=[
            CPURequest(ReqType.READ, 0x04),
            CPURequest(ReqType.READ, 0x05),
            CPURequest(ReqType.READ, 0x06),
            CPURequest(ReqType.READ, 0x07),
        ],
        mem_init={0x04:0x10, 0x05:0x20, 0x06:0x30, 0x07:0x40},
    )

    # Scenario 3: Write Hit → Dirty Bit Set
    print_simulation(
        "Scenario 3 – Write Hit and Dirty Bit",
        requests=[
            CPURequest(ReqType.READ,  0x08),
            CPURequest(ReqType.WRITE, 0x09, data=0xFF),
            CPURequest(ReqType.READ,  0x09),
        ],
        mem_init={0x08:0xAA, 0x09:0xBB, 0x0A:0xCC, 0x0B:0xDD},
    )

    # Scenario 4: Write Miss → Allocate-on-Write
    print_simulation(
        "Scenario 4 – Write Miss (Allocate-on-Write)",
        requests=[CPURequest(ReqType.WRITE, 0x10, data=0x55)],
        mem_init={0x10:0x00, 0x11:0x00, 0x12:0x00, 0x13:0x00},
    )

    # Scenario 5: Dirty Block Eviction (WRITE_BACK → ALLOCATE)
    print_simulation(
        "Scenario 5 – Dirty Eviction (WRITE_BACK → ALLOCATE)",
        requests=[
            CPURequest(ReqType.READ,  0x00),               # load idx=0, tag=0
            CPURequest(ReqType.WRITE, 0x01, data=0x99),    # dirty it
            CPURequest(ReqType.READ,  0x40),               # idx=0 tag=4 → conflict → WB
        ],
        mem_init={
            0x00:0x11, 0x01:0x22, 0x02:0x33, 0x03:0x44,
            0x40:0xAA, 0x41:0xBB, 0x42:0xCC, 0x43:0xDD,
        },
    )

    # Scenario 6: Mixed Stress Test
    print_simulation(
        "Scenario 6 – Mixed Read/Write Stress Test",
        requests=[
            CPURequest(ReqType.READ,  0x04),
            CPURequest(ReqType.WRITE, 0x05, data=0xBE),
            CPURequest(ReqType.READ,  0x08),
            CPURequest(ReqType.READ,  0x04),               # hit
            CPURequest(ReqType.WRITE, 0x08, data=0xEF),   # hit (dirty)
            CPURequest(ReqType.READ,  0x0C),
            CPURequest(ReqType.READ,  0x44),               # idx=1 conflict → dirty evict
        ],
        mem_init={
            0x04:0x10, 0x05:0x20, 0x06:0x30, 0x07:0x40,
            0x08:0x50, 0x09:0x60, 0x0A:0x70, 0x0B:0x80,
            0x0C:0x90, 0x0D:0xA0, 0x0E:0xB0, 0x0F:0xC0,
            0x44:0xDE, 0x45:0xAD, 0x46:0xBE, 0x47:0xEF,
        },
    )

    print(f"\n{'━'*72}")
    print("  ✓  All scenarios complete.")
    print(f"{'━'*72}\n")
