import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class car:

    # colour
    # length
    # velocity (assumed constant)
    # x, y, pose

    def __init__(self, length=2.3 , velocity=5, disturb=0, x=0, y=0, pose=0):

        # constructor !

        self.__length = length
        self.__velocity = velocity
        self.__x = x
        self.__y = y
        self.__pose = pose
        self.__disturb = disturb

    def move (self, steering_angle, dt):
       # simulate the motion (trajectory) of the car
       # from time t=0 to time t=0+dt starting from
       # z_initial = [self.x, self.y, self.pose]


        def bicycle_model(t, z):
            θ = z[2]

            return [self.__velocity * np.cos (θ),
                    self.__velocity * np.sin (θ),
                    self.__velocity * np.tan (steering_angle + self.__disturb ) / self.__length]

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

    def __init__(self, kp, kd, ts):
        """
        constuctor for PIDController

        :param kp:
        :param ki:
        :param kd:
        :param ts:
        """
        self.__kp = kp
        self.__kd = kd / ts
        self.__ts = ts
        self.__previous_error = None    # "not defined yet" (None)
        self.__sum_errors = 0.

    def control(self, y, y_set_point=0):
        error = y_set_point - y
        control_action =  self.__kp *  error

        if self.__previous_error is not None:
            control_action += self.__kd * (error - self.__previous_error)

        self.__previous_error = error

        return control_action

#---------------------------------------------------------------------

t_sampling = 0.025
num_points = 2000

t_span = t_sampling * np.arange(num_points + 1)


pidA  = PIDController (kp=0.05, kd=0.05, ts=0.025)
car_A = car(x=0, y=0.3, pose=0)

y_cacheA = np.array([car_A.y()]) # we inserted the correct (first) value of y
                                   # into the cache (y cache)
x_cacheA = np.array([car_A.x()])

for k in range(num_points):
    control_actionA = pidA.control(y = car_A.y())
    car_A.move(control_actionA   , t_sampling)
    y_cacheA = np.append(y_cacheA, car_A.y())
    x_cacheA = np.append(x_cacheA, car_A.x())

#resimulating with increased kd

pidB = PIDController(kp=0.05, kd=0.1, ts=0.025)
car_B = car(x=0, y=0.3, pose=0)

y_cacheB = np.array([car_B.y()])
x_cacheB = np.array([car_B.x()])

for k in range(num_points):
    control_actionB = pidB.control(y=car_B.y())
    car_B.move(control_actionB   , t_sampling)
    y_cacheB = np.append(y_cacheB, car_B.y())
    x_cacheB = np.append(x_cacheB, car_B.x())


#resimulating with increased kd again

pidC = PIDController(kp=0.05, kd=0.2, ts=0.025)
car_C = car(x=0, y=0.3, pose=0)

y_cacheC = np.array([car_C.y()])
x_cacheC = np.array([car_C.x()])

for k in range(num_points):
    control_actionC = pidC.control(y=car_C.y())
    car_C.move(control_actionC   , t_sampling)
    y_cacheC = np.append(y_cacheC, car_C.y())
    x_cacheC = np.append(x_cacheC, car_C.x())

#resimulating with increased kd again

pidD = PIDController(kp=0.05, kd=0.5, ts=0.025)
car_D = car(x=0, y=0.3, pose=0)

y_cacheD = np.array([car_D.y()])
x_cacheD = np.array([car_D.x()])

for k in range(num_points):
    control_actionD = pidD.control(y=car_D.y())
    car_D.move(control_actionD   , t_sampling)
    y_cacheD = np.append(y_cacheD, car_D.y())
    x_cacheD = np.append(x_cacheD, car_D.x())


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


plt.plot(x_cacheA, y_cacheA, label="Kd=0.05")
plt.plot(x_cacheB, y_cacheB, label="Kd=0.1")
plt.plot(x_cacheC, y_cacheC, label="Kd=0.2")
plt.plot(x_cacheD, y_cacheD, label="Kd=0.5")
plt.xlabel('lineal position, x(m)')
plt.ylabel('lateral position, y(m)')
plt.suptitle('Trajectory of car', fontsize=16)
plt.legend(loc="upper left")
plt.grid()
plt.show()

