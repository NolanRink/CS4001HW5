from neuron import h
from ballandstick import BallAndStick

# Initialize MPI support before using ParallelContext.
h.nrnmpi_init()
pc = h.ParallelContext()

class Ring:
    """
    A network of N Ball-and-Stick cells arranged in a ring.
    Each cell n makes an excitatory synapse onto cell n+1,
    with the last cell connecting back to the first.
    """
    def __init__(self, N=5, stim_w=0.04, stim_t=9, stim_delay=1, syn_w=0.01, syn_delay=5, r=50):
        self._N = N
        self.set_gids()
        self._syn_w = syn_w
        self._syn_delay = syn_delay
        self._create_cells(r)
        self._connect_cells()
        # Only the process that owns cell with gid 0 gets the stimulus.
        if pc.gid_exists(0):
            self._netstim = h.NetStim()
            self._netstim.number = 1
            self._netstim.start = stim_t
            self._nc = h.NetCon(self._netstim, pc.gid2cell(0).syn)
            self._nc.delay = stim_delay
            self._nc.weight[0] = stim_w
            
    def set_gids(self):
        """Assign cell gids round-robin across available MPI hosts."""
        self.gidlist = list(range(pc.id(), self._N, pc.nhost()))
        for gid in self.gidlist:
            pc.set_gid2node(gid, pc.id())
            
    def _create_cells(self, r):
        self.cells = []
        for i in self.gidlist:
            theta = i * 2 * h.PI / self._N
            cell = BallAndStick(i, h.cos(theta) * r, h.sin(theta) * r, 0, theta)
            self.cells.append(cell)
        for cell in self.cells:
            pc.cell(cell._gid, cell._spike_detector)
            
    def _connect_cells(self):
        """Connect each cell's output synapse to the next cell's synapse in a ring."""
        for target in self.cells:
            source_gid = (target._gid - 1 + self._N) % self._N
            nc = pc.gid_connect(source_gid, target.syn)
            nc.weight[0] = self._syn_w
            nc.delay = self._syn_delay
            target._ncs.append(nc)
