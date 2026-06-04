import numpy as np
from lagrlib.particles.larvae import ParticleState

class MovementEngine:
    def __init__(self, components = None):
        self.components = components if components is not None else []
    
    def __call__(self, t, y, n_particles):
        x = y[:n_particles]
        y_ = y[n_particles:2*n_particles]
        z = y[2*n_particles:]

        state = ParticleState(x, y_, z, age=None)
        
        u_total = np.zeros_like(x)
        v_total = np.zeros_like(x)
        w_total = np.zeros_like(x)

        for component in self.components:

            u, v, w = component.velocity(state, t)

            u_total += u
            v_total += v
            w_total += w

        return np.concatenate([
            u_total,
            v_total,
            w_total
        ])
    
    def add_component(self, component):
        self.components.append(component)