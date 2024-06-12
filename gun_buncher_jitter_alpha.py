"""
calculate jitter from 1D model
uses 1D fieldmap for gun and buncher

extract derivative first, for a reference point
then given any jitter, calculate TOA jitter
"""
import numpy as np
import jax.numpy as jnp
from jax import vmap, grad, jit, jacrev, jacfwd
from diffrax import diffeqsolve, ODETerm, SaveAt, LinearInterpolation, Tsit5, Dopri5, Dopri8
from scipy.constants import m_e, mu_0, epsilon_0, c, pi, e
from rich import print


dat_gun_z, dat_gun_Ez = np.loadtxt("gun1Dmap.txt",skiprows=1, unpack=True)
dat_buncher_z, dat_buncher_Ez = np.loadtxt("buncher1D.txt",skiprows=1, unpack=True)


dat_gun_z  = jnp.asarray(dat_gun_z)
dat_gun_Ez = jnp.asarray(dat_gun_Ez)
dat_gun_z  = dat_gun_z - dat_gun_z.min()    
print(dat_gun_z.min(),dat_gun_z.max())
ez_gun_interpl = LinearInterpolation(ts=dat_gun_z, ys=dat_gun_Ez)

dat_buncher_z  = jnp.asarray(dat_buncher_z)
dat_buncher_Ez = jnp.asarray(dat_buncher_Ez)
dat_buncher_z  = dat_buncher_z - dat_buncher_z.min()    
print(dat_buncher_z.min(),dat_buncher_z.max())
ez_buncher_interpl = LinearInterpolation(ts=dat_buncher_z, ys=dat_buncher_Ez)
print("- "*20)


def rf_dynamics_coskz_jax(z, y, args):
    """
    1D dynamics with custom field 
    field from cos(kz)
    """
    phi, gamma = y
    f, alpha = args
    
    k = 2 * pi * f / c
    # alpha = e * E0 / (2 * m * c * c * k)
    
    d_phi = k * (gamma / jnp.sqrt(gamma**2 - 1) - 1 )
    d_gamma = 2 * alpha * k * jnp.cos(k*z)  * jnp.sin(phi + k * z)
    # d_y = (d_phi, d_gamma)
    return d_phi, d_gamma
    
    
def rf_dynamics_fieldmap_jax(z, y, args):
    """
    1D dynamics with custom field 
    field from ez_interpl
    """
    phi, gamma = y
    f, alpha, ez_interpl = args
    
    k = 2 * pi * f / c
    # alpha = e * E0 / (2 * m * c * c * k)
    
    d_phi = k * (gamma / jnp.sqrt(gamma**2 - 1) - 1 )
    d_gamma = 2 * alpha * k * ez_interpl.evaluate(z)  * jnp.sin(phi + k * z)
    # d_y = (d_phi, d_gamma)
    return d_phi, d_gamma

def main():
 
    ####################### 
    ######   coskz   ######
    ####################### 

    f = 2.856e9
    wl = c / f
    k = 2 * pi * f / c
    
    phi0 = 30.0 / 180 * pi
    gamma0   = 1 + 0.5 / 0.511e6
    E0 = 90.0e6 
    alpha0 = e * E0 / (2 * m_e * c * c * k)
    print(f"{alpha0 = }")
    ref_point = jnp.array([phi0, alpha0])
    # parameters for integration
    t0 = 0
    t1 = 3 / 4 * wl
    dt0 = 1e-5 * wl

    # define parameters for solver
    term = ODETerm(rf_dynamics_coskz_jax)
    solver =  Dopri8()
    solver =  Tsit5()
    saveat = SaveAt(ts=jnp.linspace(t0, t1, 1000))
        
    def get_final_phase_and_gamma_gun(X):
        """
        phi0, alpha0 = X[0], X[1]
        return final phase (deg) and final gamma
        """
        sol = diffeqsolve(term, solver, t0, t1, dt0, (X[0], gamma0), args=(f, X[1]), saveat=saveat,max_steps= 1000000 )
        return jnp.asarray([sol.ys[0][-1], sol.ys[1][-1]])
        
    phi_gamma_final_ref = get_final_phase_and_gamma_gun(ref_point)
    jacobian_matrix = jacrev(get_final_phase_and_gamma_gun)(ref_point)
    print("- "*20)
    print(repr(phi_gamma_final_ref))
    print(repr(jacobian_matrix))
    
       
    ##################### 
    ######   Gun   ######
    ##################### 

    f = 2.856e9
    wl = c / f
    k = 2 * pi * f / c
    
    phi0 = 70 / 180 * pi
    gamma0   = 1 + 0.5 / 0.511e6
    E0 = 90.0e6 
    alpha0 = e * E0 / (2 * m_e * c * c * k)
    print(f"{alpha0 = }")
    ref_point = jnp.array([phi0, alpha0])
    # parameters for integration
    t0 = dat_gun_z.min()
    t1 = dat_gun_z.max()
    dt0 = 1e-5 * wl

    # define parameters for solver
    term = ODETerm(rf_dynamics_fieldmap_jax)
    solver =  Dopri8()
    solver =  Tsit5()
    saveat = SaveAt(ts=jnp.linspace(t0, t1, 1000))
        
    def get_final_phase_and_gamma_gun(X):
        """
        phi0, alpha0 = X[0], X[1]
        return final phase (deg) and final gamma
        """
        sol = diffeqsolve(term, solver, t0, t1, dt0, (X[0], gamma0), args=(f, X[1], ez_gun_interpl), saveat=saveat,max_steps= 1000000 )
        return jnp.asarray([sol.ys[0][-1], sol.ys[1][-1]])
        
    phi_gamma_final_ref = get_final_phase_and_gamma_gun(ref_point)
    jacobian_matrix = jacrev(get_final_phase_and_gamma_gun)(ref_point)
    print("- "*20)
    print(repr(phi_gamma_final_ref))
    print(repr(jacobian_matrix))



    ##################### 
    ###### buncher ######
    ##################### 


    phi0     = 176 / 180 * pi
    gamma0   = phi_gamma_final_ref[1]
    E0       = 25.0e6    
    alpha0 = e * E0 / (2 * m_e * c * c * k)
    print(f"{alpha0 = }")
    ref_point = jnp.array([phi0, alpha0, gamma0])
    
    # parameters for integration
    t0 = dat_buncher_z.min()
    t1 = dat_buncher_z.max()
    dt0 = 1e-5 * wl

    # define parameters for solver
    term = ODETerm(rf_dynamics_fieldmap_jax)
    solver =  Dopri8()
    solver =  Tsit5()
    saveat = SaveAt(ts=jnp.linspace(t0, t1, 1000))
        
    def get_final_phase_and_gamma_buncher(X):
        """
        phi0, alpha0, gamma0 = X[0], X[1], X[2]
        return final phase (deg) and final gamma
        """
        sol = diffeqsolve(term, solver, t0, t1, dt0, (X[0], X[2]), args=(f, X[1], ez_buncher_interpl), saveat=saveat,max_steps= 1000000 )
        return jnp.asarray([sol.ys[0][-1], sol.ys[1][-1]])
        
    phi_gamma_final_ref = get_final_phase_and_gamma_buncher(ref_point)
    jacobian_matrix = jacrev(get_final_phase_and_gamma_buncher)(ref_point)
    print("- "*20)
    print(repr(phi_gamma_final_ref))
    print(repr(jacobian_matrix))
    
        
if __name__ == "__main__":
    main()