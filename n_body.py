import numpy as np
import matplotlib.pyplot as plt

def get_acceleration(pos, mass, softening):
	"""
	Calculates the acceleration on each particle
	pos: N x 3 matrix of positions
	mass: N x 1 vector of masses
	softening: softening length
	a: N x 3 matrix of accelerations
	"""
	x = pos[:,0:1]
	y = pos[:,1:2]
	z = pos[:,2:3]
	
	dx = x.T - x
	dy = y.T - y
	dz = z.T - z
	
	inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)**(-1.5)
	
	ax = G * (dx * inv_r3) @ mass
	ay = G * (dy * inv_r3) @ mass
	az = G * (dz * inv_r3) @ mass
	
	a = np.hstack((ax,ay,az))
	
	return a

def get_energy(pos, vel, mass):
	"""
	Get kinetic energy (KE) and potential energy (PE) of simulation
	pos: N x 3 matrix of positions
	vel: N x 3 matrix of velocities
	mass: N x 1 vector of masses
	KE: total kinetic energy of the system
	PE: total potential energy of the system
	"""

	KE = 0.5 * np.sum(np.sum(mass * vel**2 ))

	x = pos[:,0:1]
	y = pos[:,1:2]
	z = pos[:,2:3]

	dx = x.T - x
	dy = y.T - y
	dz = z.T - z

	inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
	inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]

	PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))
	
	return KE, PE

if __name__ == "__main__":
    N         = 100     # Number of particles
    t         = 0       # current time of the simulation
    t_finish  = 10.0    # time at which simulation ends
    dt        = 0.01    # timestep
    softening = 0.1     # softening length
    G         = 1.0     # set Newton's Gravitational Constant as 1.0
    plotting  = True    # if True, plotting is turned on

    np.random.seed(17)            # set the random number generator seed

    mass = 20.0*np.ones((N,1))/N  # total mass of each particle is 20
    pos  = np.random.randn(N,3)
    vel  = np.random.randn(N,3)

    vel -= np.mean(mass * vel,0) / np.mean(mass) 	# Convert to Center-of-Mass frame

    acc = get_acceleration(pos, mass, softening)
    KE, PE  = get_energy(pos, vel, mass)

    Nt = int(np.ceil(t_finish/dt))

    # save energies, particle orbits for plotting trails
    pos_save = np.zeros((N,3,Nt+1))
    pos_save[:,:,0] = pos
    KE_save = np.zeros(Nt+1)
    KE_save[0] = KE
    PE_save = np.zeros(Nt+1)
    PE_save[0] = PE
    t_all = np.arange(Nt+1)*dt

    fig = plt.figure(figsize=(4,5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2,0])
    ax1.set_facecolor('white')
    ax2 = plt.subplot(grid[2,0])

    for i in range(Nt):
        vel += acc * dt/2.0
        pos += vel * dt
        acc = get_acceleration(pos, mass, softening)
        vel += acc * dt/2.0
        t += dt
        KE, PE  = get_energy(pos, vel, mass)
        
        # save energies, positions for plotting trail
        pos_save[:,:,i+1] = pos
        KE_save[i+1] = KE
        PE_save[i+1] = PE

        # plot in real time
        if plotting or (i == Nt-1):
            plt.sca(ax1)
            plt.cla()
            xx = pos_save[:,0,max(i-50,0):i+1]
            yy = pos_save[:,1,max(i-50,0):i+1]
            plt.scatter(xx,yy,s=1,color=[.7,.7,1])
            plt.scatter(pos[:,0],pos[:,1],s=10,color='navy')
            ax1.set(xlim=(-2, 2), ylim=(-2, 2))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([-2,-1,0,1,2])
            ax1.set_yticks([-2,-1,0,1,2])
            
            plt.sca(ax2)
            plt.cla()
            plt.scatter(t_all,KE_save,color='red',s=1,label='KE')
            plt.scatter(t_all,PE_save,color='blue',s=1,label='PE')
            plt.scatter(t_all,KE_save+PE_save,color='black',s=1,label='Total E')
            ax2.set(xlim=(0, t_finish), ylim=(-300, 300))
            ax2.set_aspect(0.007)

            plt.sca(ax2)
            plt.xlabel('time')
            plt.ylabel('energy')
            ax2.legend(loc='upper right')
            
            plt.pause(0.00001)
            
    plt.show()