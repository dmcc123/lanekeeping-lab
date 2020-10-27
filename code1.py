import numpy as np      #initialise numpy for advanced functions

from scipy.integrate import solve_ivp   #initialise integration function from scipy library

import matplotlib.pyplot as plt     #initialise matplot for visualising results


# x' = v * cos (θ)
# y' = v * sin (θ)
# θ' = v * tan (u) / L

# state: z = [x, y, θ]
# input: u

v = 5    # 5 m/s
L = 2.3  # 2.3m

u = -2 * np.pi/180.  #2 degrees in radians

def system_dynamics(t, z):
    θ = z[2]    # z[2] points to 3rd position in list (0, 1, 2)

    #bicycle model

    return [v * np.cos(θ),
            v * np.sin(θ),
            v * np.tan(u) / L]

t_initial = 0   # start time of simulation
t_final = 2   # end time of simulation

# the inital condition is:
# z(0) = [x(0), y(0), θ(0)]

z_initial= [0., 0.3, 5]   # here we define our initial conditions

solution = solve_ivp(system_dynamics,
                     [t_initial,t_final],
                     z_initial,
                     t_eval = np.linspace (0, t_final, 1000)) # this line gives a more high quality solution

times = solution.t
x_trajectory = solution. y[0]
y_trajectory = solution. y[1]
θ_trajectory = solution. y[2]

print(" ")
print("times = ", times)
print(" ")
print("x(t) = ", x_trajectory)
print(" ")
print(" ")
print("y(t) = ", y_trajectory)
print(" ")
print(" ")
print("θ(t) = ", θ_trajectory)
print(" ")

print (solution.y)  # print our solution to the console


plt.plot(times, y_trajectory)
plt.suptitle('y(t) vs time', fontsize=16)
plt.xlabel('time (s)')
plt.ylabel('lateral position, y(m)')
plt.grid()
plt.show()

plt.plot(times, x_trajectory)
plt.suptitle('x(t) vs time', fontsize=16)
plt.xlabel('time (s)')
plt.ylabel('lateral position, x(m)')
plt.grid()
plt.show()

plt.plot(times, θ_trajectory)
plt.suptitle('θ(t) vs time', fontsize=16)
plt.xlabel('time (s)')
plt.ylabel('angle,θ(rad)')
plt.grid()
plt.show()

plt.plot(x_trajectory, y_trajectory)
plt.suptitle('Trajectory of car', fontsize=16)
plt.xlabel('lateral position, x(m)')
plt.ylabel('lateral position, y(m)')
plt.grid()
plt.show()