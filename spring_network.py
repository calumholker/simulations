import numpy as np
import matplotlib.pyplot as plt

def sub2ind(array_shape, i, j):
    """
    Converts index (i,j) of 2D array to the index (idx) of the equivalent linear array
    eg, 3x3 matrix: 1,1 --> 4
    """
    idx = i*array_shape[1] + j
    return idx

def get_acceleration(pos, ci, cj, springL, spring_coeff, gravity):
    """
    Calculates acceleration between two nodes due to Hooke's law and gravity
    pos: N x 2 matrix of positions
    ci: index of first node in pos
    cj: index of second node in pos
    springL: springs natural length
    spring_coeff: spring coefficient
    gravity: gravitational constant
    """
    acc = np.zeros(pos.shape)
	
    separation_vector = pos[ci,:] - pos[cj,:]
    separation = np.linalg.norm(separation_vector, axis = 1)
    dL = separation - springL
    ax = - spring_coeff * dL * separation_vector[:,0] / separation
    ay = - spring_coeff * dL * separation_vector[:,1] / separation
    np.add.at(acc[:,0], ci, ax)
    np.add.at(acc[:,1], ci, ay)
    np.add.at(acc[:,0], cj, -ax)
    np.add.at(acc[:,1], cj, -ay)

    acc[:,1] += gravity

    return acc
	
def applyBoundary(pos, vel, boxsize):
    """
    Reverses velocity of node if position is outside the confines of the box
    pos: N x 2 matrix of positions
    vel: N x 2 matrix of velocities
    """
    for d in range(0,2):
        is_out = np.where(pos[:,d] < 0)
        pos[is_out, d] *= -1 
        vel[is_out, d] *= -1 
        
        is_out = np.where(pos[:,d] > boxsize)
        pos[is_out, d] *= -1 
        vel[is_out, d] *= -1 
            
    return (pos, vel)	

if __name__== "__main__":
    N         = 5      # number of nodes in single linear dimension
    t         = 0      # current time of the simulation
    dt        = 0.1    # timestep
    Nt        = 400    # number of timesteps
    spring_coeff = 40  # spring coefficient
    gravity   = -0.1   # strength of gravity
    plotting = True    # if True, plotting is turned on

    boxsize = 3
    xlin = np.linspace(1,2,N)
    x, y = np.meshgrid(xlin, xlin)
    x = x.flatten()
    y = y.flatten()

    pos = np.vstack((x,y)).T
    vel = np.zeros(pos.shape)
    acc = np.zeros(pos.shape)

    np.random.seed(17)
    vel += 0.01*np.random.randn(N**2,2) # add some random noise to make the simulation more interesting

    ci = [] # variables storing all spring connections
    cj = []

    # horizontal connections
    for r in range(0,N):
        for c in range(0,N-1):
            idx_i = sub2ind([N, N], r, c)
            idx_j = sub2ind([N, N], r, c+1)
            ci.append(idx_i)
            cj.append(idx_j)

    # vertical connections
    for r in range(0,N-1):
        for c in range(0,N):
            idx_i = sub2ind([N, N], r, c)
            idx_j = sub2ind([N, N], r+1, c)
            ci.append(idx_i)
            cj.append(idx_j)	
    
    # diagonal right connections
    for r in range(0,N-1):
        for c in range(0,N-1):
            idx_i = sub2ind([N, N], r, c)
            idx_j = sub2ind([N, N], r+1, c+1)
            ci.append(idx_i)
            cj.append(idx_j)	
    
    # diagonal left connections
    for r in range(0,N-1):
        for c in range(0,N-1):
            idx_i = sub2ind([N, N], r+1, c)
            idx_j = sub2ind([N, N], r, c+1)
            ci.append(idx_i)
            cj.append(idx_j)

    springL = np.linalg.norm( pos[ci,:] - pos[cj,:], axis = 1)

    fig = plt.figure(figsize=(4,4), dpi=80)
    ax = fig.add_subplot(111)

    for i in range(Nt):
        vel += acc * dt/2.0
        pos += vel * dt
        pos, vel = applyBoundary(pos, vel, boxsize)
        acc = get_acceleration(pos, ci, cj, springL, spring_coeff, gravity)
        vel += acc * dt/2.0
        t += dt
        
        # plot in real time
        if plotting or (i == Nt-1):
            plt.cla()
            plt.plot(pos[[ci, cj],0],pos[[ci, cj],1],color='blue')
            plt.scatter(pos[:,0],pos[:,1],s=10,color='blue')
            ax.set(xlim=(0, boxsize), ylim=(0, boxsize))
            ax.set_aspect('equal', 'box')
            ax.set_xticks([0,1,2,3])
            ax.set_yticks([0,1,2,3])
            plt.pause(0.001)