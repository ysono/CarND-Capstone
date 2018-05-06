import rospy
from lowpass import LowPassFilter
from pid import PID
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

PID_SPDCTL_P = 0.9
PID_SPDCTL_I = 0.0009
PID_SPDCTL_D = 1.75


PID_ACC_P = 0.4
PID_ACC_I = 0.05
PID_ACC_D = 0.4

LPF_ACCEL_TAU = 0.2

class Controller(object):
    def __init__(self, sampling_rate,
                 vehicle_mass, fuel_capacity,
                 brake_deadband, decel_limit, accel_limit, wheel_radius,
                 wheel_base, steer_ratio, max_lat_accel, max_steer_angle):

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.sampling_rate = sampling_rate

        tau = LPF_ACCEL_TAU # 1/(2pi*tau) = cutoff frequency
        ts = 1./self.sampling_rate # Sample time

        rospy.logwarn('TwistController: Sampling rate = ' + str(self.sampling_rate))
        rospy.logwarn('TwistController: ts  = ' + str(ts))

        self.prev_vel = 0.0
        self.current_accel = 0.0
        self.acc_lpf = LowPassFilter(tau, ts)

        self.brake_torque_const = (self.vehicle_mass + self.fuel_capacity * GAS_DENSITY) * self.wheel_radius

        # Initialise PID controller for speed control
        self.pid_spd_ctl = PID(PID_SPDCTL_P, PID_SPDCTL_I, PID_SPDCTL_D,
                                  self.decel_limit, self.accel_limit)

        # second controller to get throttle signal between 0 and 1
        self.accel_pid = PID(PID_ACC_P, PID_ACC_I, PID_ACC_D, 0.0, 0.8)

        # Initialise Yaw controller for steering control
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        # self.last_time = rospy.get_time()

    def reset(self):
        #self.accel_pid.reset()
        self.pid_spd_ctl.reset()

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        """Returns (throttle, brake, steer), or None"""

        throttle, brake, steering = 0.0, 0.0, 0.0

        if not dbw_enabled:
            self.reset()
            return None

        sampling_time = 1./self.sampling_rate
        vel_error = linear_vel - current_vel

        # calculate current acceleration and smooth using lpf
        accel_temp = self.sampling_rate * (self.prev_vel - current_vel)
        # update
        self.prev_vel = current_vel
        self.acc_lpf.filt(accel_temp)
        self.current_accel = self.acc_lpf.get()

        # use velocity controller compute desired accelaration
        desired_accel = self.pid_spd_ctl.step(vel_error, sampling_time)

        if desired_accel > 0.0:
            if desired_accel < self.accel_limit:
                throttle = self.accel_pid.step(desired_accel - self.current_accel, sampling_time)
            else:
                throttle = self.accel_pid.step(self.accel_limit - self.current_accel, sampling_time)
            brake = 0.0
        else:
            throttle = 0.0
            # reset just to be sure
            self.accel_pid.reset()
            if abs(desired_accel) > self.brake_deadband:
                # don't bother braking unless over the deadband level
                # make sure we do not brake to hard
                if abs(desired_accel) > abs(self.decel_limit):
                    brake = abs(self.decel_limit) * self.brake_torque_const
                else:
                    brake = abs(desired_accel) * self.brake_torque_const

        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        # throttle = self.throttle_controller.step(vel_error, sample_time)
        # brake = 0
        #
        # if linear_vel == 0. and current_vel < 0.1:
        #     throttle = 0
        #     brake = 400 # N*m - to hold the car in place if we are stopped at a light. Acceleration - 1m/s^2
        #
        # elif throttle < .1 and vel_error < 0:
        #     throttle = 0
        #     decel = max(vel_error, self.decel_limit)
        #     brake = abs(decel) * self.vehicle_mass * self.wheel_radius # Torque N*m

        return throttle, brake, steering
