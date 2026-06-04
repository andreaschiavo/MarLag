import numpy as np
import lagrlib.myio.velocity_fields as vf

class MovementComponent():
    def velocity(self, y, t):
        raise NotImplementedError("velocity method must be implemented by subclass")

class HorizontalAdvection(MovementComponent):        
    def velocity(self, state, t):
        points = np.column_stack((
            state.x, state.y, state.z
        )
            )
        u = vf.interp_U(points) / np.cos(np.radians(state.y))
        v = vf.interp_V(points)
        
        w = np.zeros_like(u)  # No vertical velocity in horizontal advection
        return u, v, w

class VerticalAdvection(MovementComponent):
    
    def velocity(self, state, t):
        points = np.column_stack((
            state.x, state.y, state.z
        )
            )
        w = vf.interp_W(points)
        
        u = np.zeros_like(w)  # No horizontal velocity in vertical advection
        v = np.zeros_like(w)
        return u, v, w

class Diffusion(MovementComponent):

    def __init__(self, K=10.0, dt=86400):
        self.K = K
        self.dt = dt

    def velocity(self, state, t):

        sigma = np.sqrt(2 * self.K * self.dt)

        dx = sigma * np.random.normal(size=state.x.shape)
        dy = sigma * np.random.normal(size=state.y.shape)

        M2DEG = 1 / 111319.494

        dlat = dy * M2DEG
        dlon = dx * M2DEG / np.cos(np.deg2rad(state.y))

        # converti da displacement a velocity
        u = dlon / self.dt
        v = dlat / self.dt

        w = np.zeros_like(u)

        return u, v, w

class DVM(MovementComponent):

    def __init__(self,
                 day_depth,
                 night_depth,
                 speed):

        self.day_depth = day_depth
        self.night_depth = night_depth
        self.speed = speed

    def depth_target(self, t):
        # Semplice ciclo giorno-notte basato su una funzione sinusoidale
        s = 0.5 * (1.0 - np.cos(2.0 * np.pi * t))  # scalar tra 0 e 1
        return self.night_depth + s * (self.day_depth - self.night_depth)

    def velocity(self, state, t):

        z_target = self.depth_target(t)
        
        w = (z_target - state.z) * self.speed  # velocità proporzionale alla distanza dal target
        u = np.zeros_like(w)  # No horizontal velocity in DVM
        v = np.zeros_like(w)
        return u, v, w

