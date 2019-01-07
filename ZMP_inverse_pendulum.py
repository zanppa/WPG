# -*- coding: utf-8 -*-
"""
Copyright © 2019 Lauri Peltonen

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# Needs scipy to solve the gain values
from scipy.linalg import solve_discrete_are
import numpy as np
from numpy.linalg import inv


def create_system(Ts=5.e-3, Zc=0.8, G=9.8):
    """Create a single axis (2D) state-space presentation of the inverted pendulum model.
    
    Model is according to publication [1]
    Biped walking pattern generator allowing auxiliary ZMP control
    Shuuji Kajita et al.
    Proceedings of the 2006 IEEE/RSJ international conference on intelligent
        robots and systems
    October 2006, p. 2993-2999

    The system is in form of
    X(k+1) = A*X(k) + B*U(k)
    P(k+1) = C*X(k)
    
    where (discretized form using forward euler is)
         
        | 1       T     0    |        | 0 |
    A = | G/Zc*T  1  -G/Zc*T |    B = | 0 |    C = [ 0  0  1 ]
        | 0       0     1    |        | T |
    
    X = [ x  dx  px ]^T  (location, speed, location of ZMP)
    U = [ 0 ]  (input, speed of ZMP)
    P = [ 0 ]  (output, location of the zero moment point)
    
    Input, state and output vectors are initialized to zero.
    
    The output vector contains the zero moment point (ZMP) location in
    X axis. The state vector describes the location of center of mass
    (CoM) at the pre-defined height Zc. It is assumed to travel horizontally.
    The input is acceleration of the CoM in X direction.
    
    Parameters:
        Ts (float, > 0): Sampling time (step time)
        Zc (float, > 0): Height of the center of mass (CoM)
        G (float, > 0): Acceleration due to gravity (e.g. 9.81)
        
    Returns:
        A: State transition matrix (3x3)
        B: Input matrix (3x1)
        C: Output matrix (1x3)
        X: State vector (3x1), [X dX ddX]^T of CoM
        U: Input vector (1x1), [ddX] acceleration of CoM
        P: Output vector (1x1), [px] location of the zero moment point (ZMP)
    """
    # First single axis system
    A = np.array([[1, Ts, 0], [G/Zc*Ts, 1, -G/Zc*Ts],[0, 0, 1]])
    B = np.array([[0, 0, Ts]]).T
    C = np.array([[0, 0, 1]])
    X = np.array([[0, 0, 0]]).T     # x, dx,ddx
    U = np.array([[0]]) # Input 
    P = np.array([[0]]) # Output
    
    return (A, B, C, X, U, P)


# Optimal LQI controller designer according to
#
# Design of an optimal controller for discrete-time system
# subject to previewable demand
# Tohru Katayama
# International Journal of Control, March 1985, vol 41, no. 3, p. 677-699
#
# Also referenced in e.g. 
# General ZMP preview control for bipedal walking
# Jonghoon Park, Youngil Youm
# IEEE international conference on robotics and automation, 2007, p. 2682-2687
def create_controller(A, B, C, qex=1.0, rx=1.e-5, N=320):
    """Creates an optimal LQI controller for a state-space system.
    
    The method is done according to publication [2]
    Design of an optimal controller for discrete-time system
        subject to previewable demand
    Tohru Katayama
    International Journal of Control, March 1985, vol 41, no. 3, p. 677-699    
 
    Solves an optimal control problem by minimizing
        J = sum(i=k...inf)[ Qe*(pd(i) - p(i))^2 + (dX^T)*Qx*dX + R*dU(i)^2 ]
    
    which leads to a controller of type
        u(k) = -Gi*sum(i=0...k)[p(i) - pd(i)] - Gx*X(k) - sum(i=1...N)[G(i)*pd(k+i)]
    
    where
        pd is the desired state of the output (i.e. reference)
        p is the actual output
        Qe is the loss due to tracking error
        Qx is the loss due to incremental state (zero used here)
        R is the loss due to control
        u is the controller output
        Gi is the integrator gain
        Gx is the state control gain
        G is the preview (look-ahead) gain vector
        X is the state vector
        
    This function solves the problem and then outputs the gains of the
    optimal controller. Using this method is referenced also in [1].
    
    qex, R and N can be used to tune the output performance of the controller.
    
    Inputs:
        A: State transition matrix (n x n)
        B: Input matrix (n x r)
        C: Output matrix (1 x p)
        qex (float, > 0): Optimizer loss due to tracking error (default 1.0)
        rx (float, > 0): Optimizer loss due to control (default 1.0e-7)
        N (int, >= 0): Preview controller n. of look-ahead samples (default 320)
        
    Outputs:
        Tuple of:
            Gi: Integrator gain
            Gx: State control gain
            G: Array of look-ahead gains
    """
    assert (qex >= 0), "Controller: Qex must be positive"
    assert (rx >= 0), "Controller: Rx must be positive"
    
    rn = A.shape[0]     # Size of state matrix
    rr = B.shape[1]     # Length of input vector
    rp = C.shape[0]     # Length of output vector

    Ip = np.identity(rp)
    
    # Describe the incremental time system
    # Ã = [[Ip, CA],[0, A]] where Ip is pxp unit (identity) matrix
    # Ã is then n+p x n+p matrix
    Ai = np.zeros(shape=(rn+rp, rn+rp))
    for i in range(rp):
        Ai[i, i] = Ip[i, i] # = 1
    
    CA = np.matmul(C, A)    # p x n
    for i in range(rn):
        for j in range(rp):
            Ai[j, i+rp] = CA[j, i]
        for j in range(rn):
            Ai[i+rp, j+rp] = A[i, j]
    
    # ~B = [[CB],[B]]
    Bi = np.zeros(shape=(rp+rn, rr))
    CB = np.matmul(C, B)    # p x r
    for i in range(rp):
        for j in range(rr):
            Bi[i, j] = CB[i, j]
    for i in range(rn):
        for j in range(rr):
            Bi[i+rp, j] = B[i, j]
    
    # Qe is pxp matrix => 1x1
    # Qx is nxn matrix => 3x3
    # R is rxr matrix => 1x1
    Qe = qex * np.identity(rp)
    Qx = np.zeros(shape=(rn,rn))
    R = rx * np.identity(rr)

    
    # ~Q = [[Qe, 0], [0, Qx]] = n+p x n+p
    Qi = np.zeros(shape=(rn+rp, rn+rp))
    for i in range(rp):
        for j in range(rp):
            Qi[i, j] = Qe[i, j]
    for i in range(rn):
        for j in range(rn):
            Qi[i+rp, j+rp] = Qx[i, j]
               
    # Solve the riccati equation
    Ki = solve_discrete_are(Ai, Bi, Qi, R)
    
    # Calculate the controller gains
    
    # First some constants used in furher calculations
    mTemp = np.matmul(inv(R + np.matmul(np.matmul(Bi.T, Ki), Bi)), Bi.T)
    
    # ~I = [[Ip],[0]] = p+n, p matrix
    Ii = np.zeros(shape=(rp+rn, rp))
    for i in range(rp):
        Ii[i, i] = 1.
    
    # ~F = [[CA], [A]] = right side of Ã
    Fi = np.zeros(shape=(rn+rp,rn))
    for i in range(rn+rp):
        for j in range(rn):
            Fi[i, j] = Ai[i, j+rp]
    
    # Integral gain
    Gi = np.matmul(np.matmul(mTemp, Ki), Ii)
    
    # State feedback gain
    Gx = np.matmul(np.matmul(mTemp, Ki), Fi)
    
    # Then the look-ahead gains
    mTemp2 = inv(R + np.matmul(np.matmul(Bi.T, Ki), Bi))
    Ac = Ai - np.matmul(np.matmul(np.matmul(np.matmul(Bi, mTemp2), Bi.T), Ki), Ai)
    Xx = -np.matmul(np.matmul(Ac.T, Ki), Ii)
    
    G = [-Gi]       # Matrix rxr
    for i in range(N-1):
        mG = np.matmul(mTemp, Xx)
        Xx = np.matmul(Ac.T, Xx)
        
        G.append(mG)
        
    return (Gi, Gx, G)


def create_step_pattern(dX, dY, N, Tstep, Ts, Tend):
    """Creates a sample step pattern.
    Step length and sideways motion is defined by dX and dY, and step
    duration by Tstep. N tells how many steps will be generated.
    
    The references are zero for two step durations, before the
    steps begin. The steps look like following:
        
    X:            _______
             ____|
        ____|
        
    Y:       ____
       __   |    |    ____
         |__|    |___|
    
    Inputs:
        dX (float): Step length in X direction
        dY (float): Step distance in sideways direction (Y)
        N (int): Amount of steps to take
        Tstep (float): Duration of the step in seconds
        Ts (float): Sampling time in seconds
        Tend (float): Simulation end time in seconds
        
    Outputs:
        Tuple of
            prefx: Array of step positions in X direction
            prefy: Array of step positions in Y direction
    """
    pxref = []
    pyref = []
    xr = 0
    yr = 0
    
    steps = int(Tend / Ts)
    nstep = int(Tstep / Ts)
    nlatest = nstep       # Idle one step period
    nlast = (N+2)*nstep         # Last step to do
    
    for i in range(steps):
        if i > (nlatest + nstep):
            if i > nlast:
                yr = 0
            elif yr == 0:
                yr = -dY
            else:
                yr = -yr
                xr += dX

            nlatest = i
            
        pxref.append(xr)
        pyref.append(yr)

    return (pxref, pyref)


def calculate_controller(Gi, Gx, G, X, P, ei, pd, step):
    """Calculates the controller and outputs the control vector.
    
    Controller equation is:
    u(k) = -Gi*sum(i=0...k)[p(i) - pd(i)] - Gx*X(k) - sum(i=1...N)[G(i)*pd(k+i)]
    
    X and pd (state and reference) are assumed to be 0 for step < 0
    pd is assumed to retain its last value when step > N (after simulation time)
    
    Inputs:
        Gi: Integrator gain
        Gx: State controller gain
        G: Array of preview (look-ahead) gains
        X: Current state vector
        P: Current output vector
        ei: Error integrator value
        pd: Array of references
        step: Current simulation step number
        
    Outputs:
        Tuple of
            U: New control value
            ei: New error integrator value
    """
    # Calculate and integrate the error
    err = P - pd[step]
    ei += err[0][0]
    
    # Calculate the controller output
    # State feedback and integrator
    U = -np.matmul(Gx, X) - Gi*ei
    
    # Preview part
    steps = len(pd)
    for j, gx in enumerate(G):
        index = step + j + 1
        if index >= steps:
            index = -1      # Use last reference value after the simulation

        U = U - gx * pd[index]

    return (U, ei)


# Calculates the next state using the equations
# X(k+1) = A*x(k) + B * u(k)
# P(k+1) = C*x(k)
def calculate_state(A, B, C, X, U):
    """Calculates the state transition.
    
    Equations:
    X(k+1) = A*X(k) + B*U(k)
    P(k+1) = C*X(k)
    
    Inputs:
        A, B, C: State space representation matrices of the system
        X, U: Current state and input vectors
        
    Outputs:
        Tuple of
            Xn: New state vector
            PN: New output vector
    """
    Xn = np.matmul(A, X) + np.matmul(B, U)
    Pn = np.matmul(C, X)
    
    return (Xn, Pn)


if __name__ == "__main__":
    # Simulation parameters and cart parameters
    dt = 5.e-3  # Time step
    time_end = 9   # in seconds
    g = 9.8        # gravity, m/s^2
    z = 0.8       # Height of center of gravity
    
    steps = int(time_end / dt)
    
    (A, B, C, Xx, Ux, Px) = create_system(dt, z, g)
    Xy = np.array(Xx, copy=True)
    Uy = np.array(Ux, copy=True)
    Py = np.array(Px, copy=True)
    
    (Gi, Gx, G) = create_controller(A, B, C, 1.0, 1.e-5, int(1.6/dt))
    
    (pxref, pyref) = create_step_pattern(0.3, 0.06, 5, 1.0, dt, time_end)
    pref = [pxref, pyref]
    
    
    tplot = []  # Time
    uplotx = []
    uploty = []
    comx = []   # Center of mass X pos
    comy = []   # Center of mass Y pos
    zmpx = []   # Zero moment point X
    zmpy = []   # Zero moment point Y
    zmprefx = []
    zmprefy = []
    errx = []
    spdx = []
    
    # Error sums, required for the controller
    esum = [0, 0]
    
    # Main calculation loop
    time = 0
    for step in range(steps):
    
        (Ux, esum[0]) = calculate_controller(Gi, Gx, G, Xx, Px, esum[0], pref[0], step)
        (Uy, esum[1]) = calculate_controller(Gi, Gx, G, Xy, Py, esum[1], pref[1], step)
        
        (Xx, Px) = calculate_state(A, B, C, Xx, Ux)
        (Xy, Py) = calculate_state(A, B, C, Xy, Uy)
        
        # Store to plot variables
        tplot.append(time)
        comx.append(Xx[0][0])
        comy.append(Xy[0][0])
        zmpx.append(Px[0][0])
        zmpy.append(Py[0][0])
        zmprefx.append(pref[0][step])
        zmprefy.append(pref[1][step])
        uplotx.append(Ux[0][0])
        uploty.append(Uy[0][0])
        
        time = time + dt
        
        
    import matplotlib.pyplot as plt
    
    plt.figure(1)
    plt.clf()
    plt.plot(tplot,comx, 'b-', label='X')
    plt.plot(tplot,zmpx, 'r-', label='Px')
    plt.plot(tplot,zmprefx, 'k--', label='Py,d')
    plt.legend()
    
    plt.figure(2)
    plt.clf()
    plt.plot(tplot,comy, 'b-', label='Y')
    plt.plot(tplot,zmpy, 'r-', label='Py')
    plt.plot(tplot,zmprefy, 'k--', label='Py,d')
    plt.legend()
    
    plt.figure(3)
    plt.clf()
    plt.plot(tplot,uplotx, 'm-', label='Ux')
    plt.plot(tplot,uploty, 'g-', label='Uy')
    plt.legend()
    
    #plt.figure(4)
    #plt.clf()
    #plt.plot(G)