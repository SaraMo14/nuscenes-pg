from example.discretizer.utils import calculate_object_distance, IdleTime, Detection, Velocity, Rotation, IsTrafficLightNearby,IsZebraNearby,StopAreaNearby, PedestrianNearby, IsTwoWheelNearby,LanePosition, FrontObjects, BlockProgress, NextIntersection
from example.discretizer.discretizer_d1 import AVDiscretizerD1
from example.environment import SelfDrivingEnvironment
from pgeon.discretizer import Predicate
from enum import Enum


import numpy as np
from typing import Tuple, Union

class AVDiscretizerD2(AVDiscretizerD1):
    def __init__(self, environment: SelfDrivingEnvironment, vel_discretization = 'binary', obj_discretization = "binary", id='2a'):
        super().__init__(environment)
        self.environment = environment
        self.obj_discretization = obj_discretization
        self.vel_discretization = vel_discretization
        self.id = id
        id_to_eps = { #TODO:
            '2a': 8,
            '2b': 10            
        }
        self.eps = id_to_eps.get(self.id)
        
    ##################################
    ### Overridden Methods
    ##################################

    
    def discretize(self, state: np.ndarray, detections=None) -> Tuple[Predicate, ...]:
        predicates = super().discretize(state, detections)
        return (Predicate(IdleTime, [IdleTime(0)]), ) + predicates

    
    def str_to_state(self, state_str: str) -> Tuple[Union[Predicate, ]]:
        split_str = state_str[1:].split(' ')
        non_object_predicates = 11
        
        idle_time_str,pedestrian_str, bike_str, block_str, lane_pos_str, next_inter_str, vel_str, rot_str, stop_str, zebra_str, traffic_light_str = split_str[0:non_object_predicates]

        idle_time_predicate = idle_time_str[:-2].split('(')[1]
        n_pedestrians = pedestrian_str[:-2].split('(')[1] 
        bike_predicate = IsTwoWheelNearby[bike_str[:-2].split('(')[1]] 
        progress_predicate = BlockProgress[block_str[:-2].split('(')[1]] 
        position_predicate = LanePosition[lane_pos_str[:-2].split('(')[1]]
        intersection_predicate = NextIntersection[next_inter_str[:-2].split('(')[1]]
        mov_predicate = Velocity[vel_str[:-2].split('(')[1]]
        rot_predicate = Rotation[rot_str[:-2].split('(')[1]]
        stop_predicate = StopAreaNearby[stop_str[:-2].split('(')[1]] 
        zebra_predicate = IsZebraNearby[zebra_str[:-2].split('(')[1]] 
        traffic_light_predicate = IsTrafficLightNearby[traffic_light_str[:-2].split('(')[1]] 
        predicates = [
            Predicate(IdleTime, [IdleTime(idle_time_predicate)]),
            Predicate(PedestrianNearby, [PedestrianNearby(n_pedestrians)]),
            Predicate(IsTwoWheelNearby, [bike_predicate]),
            Predicate(BlockProgress, [progress_predicate]),
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
        idle_time, pedestrian_cross, two_wheeler, block_progress, lane_position, next_intersection, velocity, rotation, stop_sign, zebra_crossing, traffic_light, *detections = state

        #NOTE: the order of the following conditions affects the yielded Predicates, thus introducing bias. Prioritize more influent predicates.
        # Generate nearby positions considering discretization
        

        for v in self.vel_values:
            if [v] != velocity.value:
                yield idle_time,pedestrian_cross, two_wheeler, block_progress, lane_position, next_intersection, Predicate(Velocity, [v]), rotation, stop_sign, zebra_crossing, traffic_light, *detections
        
        for r in Rotation:
            if [r]!= rotation.value:
                yield idle_time,pedestrian_cross, two_wheeler, block_progress, lane_position, next_intersection, velocity, Predicate(Rotation, [r]), stop_sign, zebra_crossing, traffic_light, *detections


        for s in StopAreaNearby:
            if [s]!= stop_sign.value:
                yield idle_time,pedestrian_cross, two_wheeler, block_progress, lane_position, next_intersection, velocity,rotation, Predicate(StopAreaNearby, [s]), zebra_crossing, traffic_light, *detections
        
        for z in IsZebraNearby:
            if [z]!= zebra_crossing.value:
                yield idle_time,pedestrian_cross, two_wheeler, block_progress, lane_position, next_intersection, velocity,rotation, stop_sign, Predicate(IsZebraNearby, [z]), traffic_light, *detections
   
        for t in IsTrafficLightNearby:
            if [t]!= traffic_light.value:
                yield idle_time,pedestrian_cross, two_wheeler, block_progress, lane_position, next_intersection, velocity,rotation, stop_sign, zebra_crossing, Predicate(IsTrafficLightNearby,[ t]), *detections

        
        for l in LanePosition:
            if [l] != lane_position.value:
                yield idle_time,pedestrian_cross, two_wheeler, block_progress, Predicate(LanePosition, [l]), next_intersection, velocity,rotation, stop_sign, zebra_crossing,traffic_light, *detections

        for n in NextIntersection:
            if [n] != next_intersection.value:
                yield idle_time,pedestrian_cross, two_wheeler, block_progress, lane_position, Predicate(NextIntersection,[n]), velocity,rotation, stop_sign, zebra_crossing, traffic_light, *detections
        
        for p in PedestrianNearby.discretizations[self.obj_discretization]:
            if [p] != pedestrian_cross.value:
                yield idle_time,Predicate(PedestrianNearby, [PedestrianNearby(p, self.obj_discretization)]), two_wheeler, block_progress, lane_position, next_intersection, velocity, rotation, stop_sign, zebra_crossing, traffic_light, *detections
        
        for t in IsTwoWheelNearby:
            if [t] != two_wheeler.value:
                yield idle_time,pedestrian_cross, Predicate(IsTwoWheelNearby, [t]), block_progress, lane_position, next_intersection, velocity, rotation, stop_sign, zebra_crossing, traffic_light, *detections
        
        for b in BlockProgress:
            if [b] != block_progress.value:
                yield idle_time,pedestrian_cross, two_wheeler, Predicate(BlockProgress, [b]), lane_position, next_intersection, velocity, rotation, stop_sign, zebra_crossing, traffic_light, *detections
        
            
        # Detection classes
        front_detection_class = self.DETECTION_CLASS_MAPPING.get('CAM_FRONT', None)

        for value in Detection.discretizations[self.obj_discretization]:
            if [value] != detections[0].value:
                yield idle_time, pedestrian_cross, two_wheeler, block_progress, lane_position, lane_position, next_intersection, velocity, rotation, stop_sign, zebra_crossing, traffic_light, Predicate(front_detection_class, [front_detection_class(value, self.obj_discretization)])  

        for i in IdleTime.chunks:
            if [i] != idle_time.value:
                yield Predicate(IdleTime, [IdleTime(i)]),pedestrian_cross, two_wheeler, block_progress, lane_position, next_intersection, velocity, rotation, stop_sign, zebra_crossing, traffic_light, *detections
        

        for i in IdleTime.chunks:
            for p in PedestrianNearby.discretizations[self.obj_discretization]:
                for tw in IsTwoWheelNearby:
                    for b in BlockProgress:
                        for l in LanePosition:
                            for n in NextIntersection:
                                for v in self.vel_values:
                                    for r in Rotation:
                                        for s in StopAreaNearby:
                                            for z in IsZebraNearby:
                                                for t in IsTrafficLightNearby:
                                                    for cam in Detection.discretizations[self.obj_discretization]:
                                                            nearby_state =  Predicate(IdleTime, [IdleTime(i)]), Predicate(PedestrianNearby, [PedestrianNearby(p, self.obj_discretization)]), Predicate(IsTwoWheelNearby, [tw]), Predicate(BlockProgress, [b]), \
                                                                    Predicate(LanePosition, [l]), Predicate(NextIntersection, [n]), Predicate(Velocity, [v]), \
                                                                    Predicate(Rotation, [r]), Predicate(StopAreaNearby, [s]), Predicate(IsZebraNearby, [z]), \
                                                                    Predicate(IsTrafficLightNearby, [t]), Predicate(FrontObjects, [FrontObjects(cam, self.obj_discretization)])
                                                            
                                                            #print(f'distance: {self.distance(state, nearby_state)}')
                                                            if 1 < self.distance(state, nearby_state) < self.eps:
                                                                yield nearby_state



    def distance(self, original_pred, nearby_pred):
        '''
        Function that returns the distance between 2 states.
        '''

        o_idle_time, o_pedestrian_cross, o_two_wheeler, o_block_progress, _, _, _, _, _, _, _, *_ = original_pred
        n_idle_time, n_pedestrian_cross, n_two_wheeler, n_block_progress, _, _, _, _, _, _, _, *_ = nearby_pred

        
        distance = super().distance(original_pred[4:], nearby_pred[4:]) + int(o_idle_time.value != n_idle_time.value)

        if self.obj_discretization == 'binary':
            distance += int(o_pedestrian_cross.value != n_pedestrian_cross.value)
        else:
            distance += calculate_object_distance(o_pedestrian_cross.value[0].count, n_pedestrian_cross.value[0].count)

        return distance + int(o_two_wheeler.value !=n_two_wheeler.value) + int(o_block_progress.value !=n_block_progress.value)         
    



    #TODO: test
    def get_predicate_space(self):
        all_tuples = []
        for i in IdleTime.chunks:
            for p in PedestrianNearby.discretizations[self.obj_discretization]:
                for tw in IsTwoWheelNearby:
                    for b in BlockProgress:
                        for l in LanePosition:
                            for n in NextIntersection:
                                for v in self.vel_values: 
                                    for r in Rotation:
                                        for s in StopAreaNearby:
                                            for z in IsZebraNearby:
                                                for t in IsTrafficLightNearby:
                                                    for cam in Detection.discretizations[self.obj_discretization]:
                                                    #for right_cam in Detection.discretizations[self.obj_discretization]:
                                                        #for left_cam in Detection.discretizations[self.obj_discretization]:
                                                            all_tuples.append((i, p, tw, b, l,n,v,r,s,z,t,cam))#, left_cam))
        return all_tuples
    

    
