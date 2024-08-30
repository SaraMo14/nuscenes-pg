from example.dataset.utils import vector_angle
from example.discretizer.utils import calculate_object_distance, calculate_velocity_distance, IdleTime, IsTrafficLightNearby,IsZebraNearby, Detection, FrontObjects, Action, LanePosition, StopAreaNearby, BlockProgress, NextIntersection, Velocity, Rotation
from example.environment import SelfDrivingEnvironment
import numpy as np
from typing import Tuple, Union
import ast
from pgeon.discretizer import Discretizer, Predicate
from example.dataset.utils import create_rectangle

class AVDiscretizer(Discretizer):
    def __init__(self, environment: SelfDrivingEnvironment, vel_discretization = 'binary', obj_discretization = "binary", id = '0a'):     
        
        super().__init__()

        
        self.id = id
        self.obj_discretization = obj_discretization #'binary' (0/1) or 'multiple' (0, 1-3, 4+)
        self.vel_discretization = vel_discretization
        self.environment = environment
        
        id_to_eps = {
            '0a': 5,
            '0b': 7
        }
        self.eps = id_to_eps.get(self.id) #distance eps

        self.eps_rot = 0.3
        self.eps_vel = 0.2  
        self.eps_acc = 0.3
        self.vel_values = [Velocity.STOPPED, Velocity.MOVING] if self.vel_discretization == 'binary' else [Velocity.STOPPED, Velocity.LOW, Velocity.MEDIUM, Velocity.HIGH]
        
        self.agent_size = (1.730, 4.084) #width, length in meters

        self.state_to_be_discretized = ['x', 'y', 'velocity', 'steering_angle', 'yaw'] 
        self.state_columns_for_action = ['velocity', 'acceleration', 'steering_angle']     

        self.DETECTION_CLASS_MAPPING = {
        'CAM_FRONT': FrontObjects
        #'CAM_FRONT_LEFT': FrontLeftObjects,
        #'CAM_FRONT_RIGHT': FrontRightObjects
        }
        self.STR_TO_CLASS_MAPPING = {
            'FrontObjects': FrontObjects
            #'FrontLeftObjects': FrontLeftObjects,
            #'FrontRightObjects': FrontRightObjects
        }
    
    @staticmethod
    def is_close(a, b, eps=0.1):
        return abs(a - b) < eps

    ##################################
    ### DISCRETIZERS
    ##################################

    def discretize(self,
                   state: np.ndarray, detections=None
                   ) -> Tuple[Predicate, ...]:
        x, y, velocity, steer_angle, yaw = state 
        block_progress, lane_pos_pred  = self.discretize_position(x,y,yaw)
        mov_predicate = self.discretize_speed(velocity)
        rot_predicate = self.discretize_steering_angle(steer_angle)
        sign_predicate, zebra_predicate, traffic_light_predicate = self.discretize_stop_line(x,y,yaw)
        
        detected_predicates = self.discretize_detections(detections)
        return (Predicate(BlockProgress, [block_progress]),
                Predicate(LanePosition, [lane_pos_pred]),
                Predicate(NextIntersection, [NextIntersection.NONE]),
                Predicate(Velocity, [mov_predicate]),
                Predicate(Rotation, [rot_predicate]),
                Predicate(StopAreaNearby, [sign_predicate]),
                Predicate(IsZebraNearby,[zebra_predicate]),
                Predicate(IsTrafficLightNearby, [traffic_light_predicate]),
                *detected_predicates)
        

    def discretize_detections(self, detections):
        detected_predicates = []
        for cam_type, objects in detections.items():
            tot_count = 0
            for (category, _), count in ast.literal_eval(objects).items():
                if 'human' not in category:
                    tot_count+=count
            detection_class = self.DETECTION_CLASS_MAPPING.get(cam_type, None)
            predicate = Predicate(
                    detection_class,
                    [detection_class(tot_count, self.obj_discretization)]
            )
            detected_predicates.append(predicate)
        return detected_predicates



    def discretize_steering_angle(self, steering_angle: float)->Rotation:
        if steering_angle <= -self.eps_rot:  
            return Rotation.RIGHT
        elif steering_angle <= self.eps_rot:  
            return Rotation.FORWARD
        else:
            return Rotation.LEFT


    def discretize_position(self, x,y,yaw)-> LanePosition:
        
        block_progress, lane_position = self.environment.get_position_predicates(x,y, yaw, eps=0.3, agent_size=self.agent_size)
        
        return block_progress, lane_position
    


    def discretize_speed(self, speed) -> Velocity:
        if speed <= self.eps_vel: 
            return Velocity.STOPPED
        else:
            if self.vel_discretization == 'binary':
                return Velocity.MOVING
            else: 
                if speed <= 4.1:
                    return Velocity.LOW
                elif speed <=8.3:
                    return Velocity.MEDIUM
                else:
                    return Velocity.HIGH 
    
    def discretize_stop_line(self, x,y,yaw):
        # Create a rotated rectangle around the vehicle's current pose
        yaw_in_deg = np.degrees(-(np.pi / 2) + yaw)
        area = create_rectangle((x,y), yaw_in_deg, size=(16,20), shift_distance=10)
        
        stop_area = self.environment.is_near_stop_area(x,y,area)
        if stop_area is None:
            is_stop_nearby = StopAreaNearby.NO
        elif 'STOP_SIGN' == stop_area:
            is_stop_nearby = StopAreaNearby.STOP
        elif 'YIELD' == stop_area:
            is_stop_nearby = StopAreaNearby.YIELD
        elif 'TURN_STOP' == stop_area:
            is_stop_nearby = StopAreaNearby.TURN_STOP
        else:
            is_stop_nearby = StopAreaNearby.NO


        is_zebra_nearby = IsZebraNearby.YES  if self.environment.is_near_ped_crossing(area) else IsZebraNearby.NO
        
        is_traffic_light_nearby = IsTrafficLightNearby.YES  if self.environment.is_near_traffic_light(yaw, area) else IsTrafficLightNearby.NO

        return is_stop_nearby, is_zebra_nearby, is_traffic_light_nearby


    def assign_intersection_actions(self,trajectory, intersection_info, verbose = False):
        """
        Assigns actions based on intersection information.

        Args:
            trajectory: List containing the discretized trajectory.
            intersection_info: List storing information about intersections as.

        Returns:
            Updated trajectory with assigned actions for intersections.
        """
        for i in range(0, len(trajectory), 2):  # Access states

            for idx, action in intersection_info:
                if 2 * idx > i and 2 * idx < len(trajectory) - 1: #check if the intersection state (2*idx) comes next the current state (i)
                    state = list(trajectory[i])
                    next_intersect_idx = next((i for i, predicate in enumerate(state) if predicate.predicate.__name__ == 'NextIntersection'), None)
                    state[next_intersect_idx] = action
                    trajectory[i] = tuple(state)
                    break
            if verbose:
                    print(f'frame {int(i/2)} --> {list(trajectory[i])}')
                    if i<len(trajectory) - 1:
                        print(f'action: {self.get_action_from_id(trajectory[i+1])}')
                    else:
                        print('END')

        return trajectory
    

    @staticmethod
    def determine_intersection_action(start_position, end_position) -> NextIntersection:
        """
        Determine the action at the intersection based on positional changes.
        
        Args:
            start_position (x,y,x1,y1): vector containing starting position just before the intersection and at the beginning of the intersection.
            end_position (x,y,x1,y1): vector containin position at the intersection and just after the intersection.
        Returns:
            Action: NextIntersection.RIGHT, NextIntersection.LEFT, or NextIntersection.STRAIGHT.
        """


        x_1,y_1, x_2, y_2 = start_position
        x_n,y_n, x_n1, y_n1 = end_position

        #Calculate the movement vector from point (x1, y1) to point (x2, y2)
        pre_vector = np.array([x_2 - x_1, y_2 - y_1]) 
        post_vector = np.array([x_n1 - x_n, y_n1 - y_n]) 
        angle = vector_angle(pre_vector, post_vector)
        
        if abs(angle) < np.radians(30):
            return Predicate(NextIntersection,[NextIntersection.STRAIGHT])
        elif np.cross(pre_vector, post_vector) > 0:
            return Predicate(NextIntersection,[NextIntersection.LEFT])
        else:
            return Predicate(NextIntersection,[NextIntersection.RIGHT])

    ##################################
    




    def determine_action(self, next_state) -> Action:
        vel_t1,  acc_t1, steer_t1 = next_state
        if vel_t1 <= self.eps_vel and self.is_close(acc_t1,0,self.eps_acc):
            return Action.IDLE
    
        # determine acceleration
        if acc_t1 >= self.eps_acc and vel_t1>self.eps_vel:
            acc_action = Action.GAS
        elif acc_t1 <= -self.eps_acc and vel_t1>self.eps_vel:
            acc_action = Action.BRAKE
        else:
            acc_action = None

        # determine direction
        if steer_t1 <= -self.eps_rot:#_r:#delta_x1 > self.eps_pos_x: 
            dir_action = Action.TURN_RIGHT
        elif steer_t1 > self.eps_rot:#_l:#delta_x1 < -self.eps_pos_x:
            dir_action = Action.TURN_LEFT
        else:
            dir_action = Action.STRAIGHT

       # Combine acceleration and direction actions
        if acc_action == Action.GAS:
            if dir_action == Action.TURN_RIGHT:
                return Action.GAS_TURN_RIGHT
            elif dir_action == Action.TURN_LEFT:
                return Action.GAS_TURN_LEFT
            else:
                return Action.GAS
        elif acc_action == Action.BRAKE:
            if dir_action == Action.TURN_RIGHT:
                return Action.BRAKE_TURN_RIGHT
            elif dir_action == Action.TURN_LEFT:
                return Action.BRAKE_TURN_LEFT
            else:
                return Action.BRAKE
        elif acc_action is None:
            # Fallback to direction if no acceleration action was determined
            return dir_action

        # if no other conditions met
        return Action.STRAIGHT



    def _discretize_state_and_action(self, scene, i):
        #Given a scene from the dataset, it discretizes the current state (i) and determines the  following action.
        current_state_to_discretize = scene.iloc[i][self.state_to_be_discretized].tolist()
        current_detection_info = scene.iloc[i][self.detection_cameras]# if self.detection_cameras else None
        discretized_current_state = self.discretize(current_state_to_discretize, current_detection_info)

        next_state_for_action = scene.iloc[i+1][self.state_columns_for_action].tolist()
        action = self.determine_action(next_state_for_action)
        action_id = self.get_action_id(action)
        
        return discretized_current_state, action_id

    @staticmethod
    def get_action_id(action):
            return action.value
    
    @staticmethod
    def get_action_from_id(action_id):
        for action in Action:
            if action.value == action_id:
                return action
        raise ValueError("Invalid action ID")


    def state_to_str(self,
                     state: Tuple[Union[Predicate, ]]
                     ) -> str:
        return ' '.join(str(pred) for pred in state)

    
    def str_to_state(self, state_str: str) -> Tuple[Union[Predicate, ]]:
        split_str = state_str[1:].split(' ')
        non_object_predicates = 7
        lane_pos_str, next_inter_str, vel_str, rot_str, stop_str, zebra_str, traffic_light_str = split_str[0:non_object_predicates]
        position_predicate = LanePosition[lane_pos_str[:-2].split('(')[1]]
        intersection_predicate = NextIntersection[next_inter_str[:-2].split('(')[1]]
        mov_predicate = Velocity[vel_str[:-2].split('(')[1]]
        rot_predicate = Rotation[rot_str[:-2].split('(')[1]]
        stop_predicate = StopAreaNearby[stop_str[:-2].split('(')[1]] 
        zebra_predicate = IsZebraNearby[zebra_str[:-2].split('(')[1]] 
        traffic_light_predicate = IsTrafficLightNearby[traffic_light_str[:-2].split('(')[1]] 
        

        predicates = [
            
            Predicate(LanePosition, [position_predicate]),
            Predicate(NextIntersection, [intersection_predicate]),
            Predicate(Velocity, [mov_predicate]),
            Predicate(Rotation, [rot_predicate]),
            Predicate(StopAreaNearby, [stop_predicate]),
            Predicate(IsZebraNearby, [zebra_predicate]),
            Predicate(IsTrafficLightNearby, [traffic_light_predicate])
        ]
        

        detected_predicates = []
        for cam_detections in split_str[non_object_predicates:]:
            detection_class_str, count = cam_detections[:-2].split('(')
            detection_class = self.STR_TO_CLASS_MAPPING.get(detection_class_str, None)
            detected_predicates.append(Predicate(detection_class, [detection_class(count)]))
         
        predicates.extend(detected_predicates)
        return tuple(predicates)

    
    
    def nearest_state(self, state):
        lane_position, next_intersection, velocity, rotation, stop_sign, zebra_crossing, traffic_light, *detections = state

        #NOTE: the order of the following conditions affects the yielded Predicates, thus introducing bias. Prioritize more influent predicates.
        # Generate nearby positions considering discretization
        
        for v in self.vel_values:
            if [v] != velocity.value:
                yield lane_position, next_intersection, Predicate(Velocity, [v]), rotation, stop_sign, zebra_crossing, traffic_light, *detections
        
        for r in Rotation:
            if [r]!= rotation.value:
                yield lane_position, next_intersection, velocity, Predicate(Rotation, [r]), stop_sign, zebra_crossing, traffic_light, *detections


        for s in StopAreaNearby:
            if [s]!= stop_sign.value:
                yield lane_position, next_intersection, velocity,rotation, Predicate(StopAreaNearby, [s]), zebra_crossing, traffic_light, *detections
        
        for z in IsZebraNearby:
            if [z]!= zebra_crossing.value:
                yield lane_position, next_intersection, velocity,rotation, stop_sign, Predicate(IsZebraNearby, [z]), traffic_light, *detections
   
        for t in IsTrafficLightNearby:
            if [t]!= traffic_light.value:
                yield lane_position, next_intersection, velocity,rotation, stop_sign, zebra_crossing, Predicate(IsTrafficLightNearby,[ t]), *detections

        
        for l in LanePosition:
            if [l] != lane_position.value:
                yield Predicate(LanePosition, [l]), next_intersection, velocity,rotation, stop_sign, zebra_crossing,traffic_light, *detections

        for n in NextIntersection:
            if [n] != next_intersection.value:
                yield lane_position, Predicate(NextIntersection,[n]), velocity,rotation, stop_sign, zebra_crossing, traffic_light, *detections
        
        
            
        # Detection classes
        front_detection_class = self.DETECTION_CLASS_MAPPING.get('CAM_FRONT', None)

        for value in Detection.discretizations[self.obj_discretization]:
            if [value] != detections[0].value:
                yield lane_position, lane_position, next_intersection, velocity, rotation, stop_sign, zebra_crossing, traffic_light, Predicate(front_detection_class, [front_detection_class(value, self.obj_discretization)])  

        
        for l in LanePosition:
            for n in NextIntersection:
                for v in self.vel_values:
                    for r in Rotation:
                        for s in StopAreaNearby:
                            for z in IsZebraNearby:
                                for t in IsTrafficLightNearby:
                                    for cam in Detection.discretizations[self.obj_discretization]:
                                        nearby_state =  Predicate(LanePosition, [l]), Predicate(NextIntersection, [n]), Predicate(Velocity, [v]), \
                                                                Predicate(Rotation, [r]), Predicate(StopAreaNearby, [s]), Predicate(IsZebraNearby, [z]), \
                                                                Predicate(IsTrafficLightNearby, [t]), Predicate(FrontObjects, [FrontObjects(cam, self.obj_discretization)])#,\
                                                                #Predicate(FrontRightObjects, [FrontLeftObjects(left_cam, self.obj_discretization)])
                                        #print(f'Distance: {self.distance(state, nearby_state)}')
                                        if 1 < self.distance(state, nearby_state) < self.eps:
                                            yield nearby_state
    
    
    
    def distance(self, original_pred, nearby_pred):
        '''
        Function that returns the distance between 2 states.
        '''
        o_lane_position, o_next_intersection, o_velocity, o_rotation, o_stop_sign, o_zebra_crossing, o_traffic_light, *o_detections = original_pred
        n_lane_position, n_next_intersection, n_velocity, n_rotation, n_stop_sign, n_zebra_crossing, n_traffic_light, *n_detections = nearby_pred

        obj_distance = int(o_detections[0].value != n_detections[0].value) if self.obj_discretization == 'binary' else calculate_object_distance(o_detections[0].value[0].count, n_detections[0].value[0].count)

        vel_distance =  int(o_velocity.value != n_velocity.value) if self.vel_discretization == 'binary' else calculate_velocity_distance(o_velocity.value[0], n_velocity.value[0])
        
        distance = \
                int(o_lane_position.value != n_lane_position.value) + int(o_next_intersection.value!= n_next_intersection.value) \
                    + int(o_stop_sign.value != n_stop_sign.value) + int( o_zebra_crossing.value != n_zebra_crossing.value)  \
                    + int(o_traffic_light.value != n_traffic_light.value) + vel_distance  \
                    + int(o_rotation.value !=n_rotation.value) +  obj_distance #right_obj_distance + left_obj_distance
        return distance                      

    
    
    
    
    def all_actions(self):
        return list(Action) 
        
    #TODO: test
    def get_predicate_space(self):
        all_tuples = []
        for l in LanePosition:
            for n in NextIntersection:
                for v in self.vel_values:
                    for r in Rotation:#Rotation.FORWARD, Rotation.LEFT, Rotation.RIGHT]:
                        for s in StopAreaNearby:
                            for z in IsZebraNearby:
                                for t in IsTrafficLightNearby:
                                    for cam in Detection.discretization[self.obj_discretization]:
                                    #for right_cam in Detection.discretizations[self.obj_discretization]:
                                        #for left_cam in Detection.discretizations[self.obj_discretization]:
                                            all_tuples.append((l,n,v,r,s,z,t,cam))#right_cam, left_cam))
        return all_tuples

