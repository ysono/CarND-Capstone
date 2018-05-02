from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass,fuel_capacity,brake_deadband,decel_limit,
                accel_limit,wheel_radius,wheel_base,steer_ratio,max_lat_accel,max_steer_angle):
        
        # TODO: Implement
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        
        kp = 0.3
        ki = 0.1
        kd = 0
        mn = 0   #min throttle
        mx = 0.2 #max throttle
        self.throttle_controller = PID(kp,ki,kd,mn,mx)
        
        tau = 0.5 #1/(2pi*tau) = cutoff frequency
        ts = 0.02 #Sample time
        self.lp_filter = LowPassFilter(tau,ts) #velocity is noisy, filter out high frequency noise
        
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        
        self.last_time = rospy.get_time()


    def control(self, cur_v, dbw_enabled, linear_v, angular_v):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        rospy.loginfo("current spd: {x}".format(x=cur_v))
        rospy.loginfo("target spd: {x}".format(x=linear_v))
        
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.,0.,0.
        
        cur_v = self.lp_filter.filt(cur_v)
        
        steer = self.yaw_controller.get_steering(linear_v, angular_v, cur_v)
        
        err_v = linear_v - cur_v
        self.last_v = cur_v
        
        cur_time = rospy.get_time()
        spl_time = cur_time - self.last_time
        self.last_time = cur_time
        
        throttle = self.throttle_controller.step(err_v, spl_time)
        brake = 0
        
        if linear_v == 0. and cur_v < 0.1:
            throttle = 0.
            brake = 400   # brake torque Nm
            
        elif throttle < .1 and err_v < 0:
            throttle = 0
            decel = max(err_v,self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius 
            
        rospy.loginfo("thr:{x};brk:{y};str:{z}".format(x=throttle,y=brake,z=steer))    
            
        return throttle, brake, steer
            
            
            
            
            
            
        
        
