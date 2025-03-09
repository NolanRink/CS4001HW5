from neuron import h, gui
from neuron.units import ms, mV
from ring import Ring
import time

# Set simulation duration
sim_time = 100 * ms

# Get the ParallelContext (already initialized in ring.py)
pc = h.ParallelContext()
pc.set_maxstep(10 * ms)  # Maximum time step for parallel communication

# Record the simulation time vector (optional for further analysis)
t_vec = h.Vector().record(h._ref_t)

# Start the timer
start_time = time.time()

# Create the network ring (default: 5 cells arranged in a ring)
ring = Ring(N=5)

# Run the simulation using parallel solver
pc.psolve(sim_time)

# End the timer
end_time = time.time()
execution_time = end_time - start_time

# Only the process owning cell with gid 0 prints the execution time
if pc.id() == 0:
    print("Total execution time: {:.3f} seconds".format(execution_time))

# Synchronize processes before finishing
pc.barrier()
pc.done()
h.quit()
