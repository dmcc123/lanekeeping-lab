import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class car:

    # colour
    # length
    # velocity (assumed constant)
    # x, y, pose

    def __init__(self, length=2.3 , velocity=5 , x=0, y=0, pose=0):

        # constructor !

        self.__length = length
        self.__velocity = velocity
        self.__x = x
        self.__y = y
        self.__pose = pose

    def move (self, steering_angle, dt):
       # simulate the motion (trajectory) of the car
       # from time t=0 to time t=0+dt starting from
       # z_initial = [self.x, self.y, self.pose]


        def bicycle_model(t, z):
            θ = z[2]

            return [self.__velocity * np.cos (θ),
                    self.__velocity * np.cos (θ),
                    self.__velocity * np.tan (steering_angle + 0.0174533 ) / self.__length]

        z_initial = [self.__x, self.__y, self.__pose]
        solution = solve_ivp (bicycle_model,
                               [0,dt],
                               z_initial)
        self.__x = solution.y [0][-1]
        self.__y = solution.y [1][-1]
        self.__pose = solution.y [2][-1]

    def x(self):
        return self.__x
    def y(self):
        return self.__y
    def pose(self):
        return self.__pose
    def length(self):
        return self.__length
    def velocity(self):
        return self.__velocity


class PIDController:

    # Kp: proportional gain
    # Kd: derivative gain
    # Ki: integral gain
    # Ts sampling time

    def __init__(self, kp, ki, kd, ts):
        """
        constuctor for PIDController

        :param kp:
        :param ki:
        :param kd:
        :param ts:
        """
        self.__kp = kp
        self.__kd = kd / ts
        self.__ki = ki * ts
        self.__ts = ts
        self.__previous_error = None    # "not defined yet" (None)
        self.__sum_errors = 0.

    def control(self, y, y_set_point=0):
        error = y_set_point - y
        control_action =  self.__kp *  error

        if self.__previous_error is not None:
            control_action += self.__kd * (error - self.__previous_error)

        control_action += self.__ki * self.__sum_errors

        self.__previous_error = error

        return control_action

#---------------------------------------------------------------------

t_sampling = 0.025
num_points = 2000

t_span = t_sampling * np.arange(num_points + 1)


pidA  = PIDController (kp=0.05, ki= 0., kd=0., ts=0.025)
murphyA = car(x=0, y=0.3, pose=0)

y_cacheA = np.array([murphyA.y()]) # we inserted the correct (first) value of y
                                   # into the cache (y cache)
x_cacheA = np.array([murphyA.x()])

for k in range(num_points):
    control_actionA = pidA.control(y = murphyA.y())
    murphyA.move(control_actionA + 0.01 , t_sampling)
    y_cacheA = np.append(y_cacheA, murphyA.y())
    x_cacheA = np.append(x_cacheA, murphyA.x())

#resimulating with increased kp

pidB = PIDController(kp=1, ki=0., kd=0., ts=0.025)
murphyB = car(x=0, y=0.3, pose=0)

y_cacheB = np.array([murphyB.y()])
x_cacheB = np.array([murphyB.x()])

for k in range(num_points):
    control_actionB = pidB.control(y=murphyB.y())
    murphyB.move(control_actionB + 0.01, t_sampling)
    y_cacheB = np.append(y_cacheB, murphyB.y())
    x_cacheB = np.append(x_cacheB, murphyB.x())


#resimulating with increased kp again

pidC = PIDController(kp=10, ki=0., kd=0., ts=0.025)
murphyC = car(x=0, y=0.3, pose=0)

y_cacheC = np.array([murphyC.y()])
x_cacheC = np.array([murphyC.x()])

for k in range(num_points):
    control_actionC = pidC.control(y=murphyC.y())
    murphyC.move(control_actionC + 0.01, t_sampling)
    y_cacheC = np.append(y_cacheC, murphyC.y())
    x_cacheC = np.append(x_cacheC, murphyC.x())

#initial plots to test validity of the simulator

plt.plot(t_span, x_cacheA)
plt.xlabel('Time (s)')
plt.ylabel('lineal position, x(m)')
plt.suptitle('x(t) vs time', fontsize=16)
plt.grid()
plt.show()

plt.plot(t_span, y_cacheA)
plt.xlabel('Time (s)')
plt.ylabel('lateral position, y(m)')
plt.suptitle('y(t) vs time', fontsize=16)
plt.grid()
plt.show()

plt.plot(x_cacheA, y_cacheA)
plt.xlabel('lineal position, x(m)')
plt.ylabel('lateral position, y(m)')
plt.suptitle('Trajectory of car', fontsize=16)
plt.grid()
plt.show()

# creating the final plot with all the values we need for this exercise


plt.plot(x_cacheA, y_cacheA,  color="deepskyblue", label="Kp=0.05")
plt.plot(x_cacheB, y_cacheB,  color="orangered"  , label="Kp=1")
plt.plot(x_cacheC, y_cacheC,  color="springgreen", label="Kp=10")
plt.xlabel('lineal position, x(m)')
plt.ylabel('lateral position, y(m)')
plt.suptitle('Trajectory of car', fontsize=16)
plt.legend(loc="upper left")
plt.grid()
plt.show()

