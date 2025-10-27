# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import jax.numpy as jnp
import jaxion


class DarkMatterBenchmarks:
    """
    Benchmark a dark matter only simulation
    """

    def setup(self):
        params = {"output": {"save": False}, "time": {"end": 0.1}}
        self.sim = jaxion.Simulation(params)
        fac = 2.0 * jnp.pi / 10.0
        xx, yy, _ = self.sim.grid
        self.sim.state["psi"] = jnp.sin(fac * xx) * jnp.cos(fac * yy) + 0.0j

    def time_simulation(self):
        self.sim.run()

    def peakmem_simulation(self):
        self.sim.run()

    def time_simulation_with_save(self):
        self.sim.params["output"]["save"] = True
        self.sim.run()

    def peakmem_simulation_with_save(self):
        self.sim.params["output"]["save"] = True
        self.sim.run()


class DarkMatterGasBenchmarks:
    """
    Benchmark a dark matter + gas simulation
    """

    def setup(self):
        params = {
            "physics": {"hydro": True},
            "time": {"end": 0.1},
            "output": {"save": False},
        }
        self.sim = jaxion.Simulation(params)
        fac = 2.0 * jnp.pi / 10.0
        xx, yy, _ = self.sim.grid
        self.sim.state["psi"] = jnp.sin(fac * xx) * jnp.cos(fac * yy) + 0.0j
        self.sim.state["rho"] = jnp.ones_like(xx)

    def time_simulation(self):
        self.sim.run()

    def peakmem_simulation(self):
        self.sim.run()


class DarkMatterStarsBenchmarks:
    """
    Benchmark a dark matter + stars simulation
    """

    def setup(self):
        params = {
            "physics": {"particles": True},
            "time": {"end": 0.1},
            "output": {"save": False},
            "particles": {"num_particles": 1000},
        }
        self.sim = jaxion.Simulation(params)
        fac = 2.0 * jnp.pi / 10.0
        xx, yy, _ = self.sim.grid
        self.sim.state["psi"] = jnp.sin(fac * xx) * jnp.cos(fac * yy) + 0.0j
        self.sim.state["rho"] = jnp.ones_like(xx)
        slin = jnp.linspace(0.01, self.sim.box_size - 0.01, 10)
        xx, yy, zz = jnp.meshgrid(slin, slin, slin, indexing="ij")
        self.sim.state["pos"] = jnp.stack(
            [xx.flatten(), yy.flatten(), zz.flatten()], axis=-1
        )
        self.sim.state["vel"] = jnp.zeros((self.sim.num_particles, 3))

    def time_simulation(self):
        self.sim.run()

    def peakmem_simulation(self):
        self.sim.run()
