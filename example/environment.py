from typing import Any
from pgeon.environment import Environment
from example.discretizer.utils import BlockProgress, LanePosition
import pandas as pd    
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.map_expansion import arcline_path_utils
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
from example.dataset.utils import create_rectangle, determine_travel_alignment
from typing import Tuple, List
from shapely.geometry import Polygon, Point
import ast

class SelfDrivingEnvironment(Environment):

    def __init__(self, city = "boston-seaport", dataroot = 'example/dataset/data/sets/nuscenes'):
        
        self.city = city
        if city != 'all':
            self.nusc_map = NuScenesMap(dataroot=dataroot, map_name = city)
            self.dividers = getattr(self.nusc_map, 'road_divider') + getattr(self.nusc_map, 'lane_divider')

        #self.current_state = None

    
    def reset(self):
        pass

        
    def step():
        pass

    

    #######################
    ### RENDERING
    #######################

    
    def render_ego_influent_area(self, x,y,yaw, patch_size=20, non_geometric_layers=['road_divider', 'lane_divider'], size = (14,20), shift_distance = 10):
        
        """
        Render the ego vehicle's influent area on a map.

        :param x: x-coordinate of the ego vehicle's position.
        :param y: y-coordinate of the ego vehicle's position.
        :param yaw: Orientation of the ego vehicle in radians.
        :param patch_size: Size of the patch to render.
        :param non_geometric_layers: List of non-geometric layers to render (default includes 'road_divider' and 'lane_divider').
        :param shape: Shape of the influent area (rectangle, semi_circle or semi_ellipse).
        :param shape: (width,length) of the rectangle if shape is 'rectangle'
        :param shift_distance: Distance to shift the center of the shape in the direction of yaw.
        :param radius: Radius of the semi-circle if shape is 'semi_circle'. (radius_x, radius_y) in shape is 'semi_ellipse'

        """
        patch_box = [x,y, patch_size, patch_size]
        patch = NuScenesMapExplorer.get_patch_coord(patch_box)
        minx, miny, maxx, maxy = patch.bounds
        
        fig, ax = self.nusc_map.render_map_patch( [minx, miny, maxx, maxy], non_geometric_layers, figsize=(4, 4), render_egoposes_range=False)
            
        ax.scatter(x,y, color='red')
        yaw_in_deg =  math.degrees(-(math.pi / 2) + yaw)
        
        front_area = create_rectangle((x,y), yaw_in_deg,size, shift_distance)
        x,y = front_area.exterior.xy
        ax.plot(x,y,linewidth=0.4, color='red')

        

        plt.title('Ego Vehicle Scanning Area in The Map')
        plt.show()   
    
    
    
    
    
    
    def render_egoposes_on_fancy_map(self, map_poses:list = [], 
                                     verbose: bool = True,
                                     out_path: str = None,
                                     render_egoposes: bool = True,
                                     render_egoposes_range: bool = True,
                                     render_legend: bool = True):
        """
        Renders each ego pose of a trajectory on the map.
        
        :param map_poses: List of poses on the map.
        :param verbose: Whether to show status messages and progress bar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param render_egoposes: Whether to render ego poses.
        :param render_egoposes_range: Whether to render a rectangle around all ego poses.
        :param render_legend: Whether to render the legend of map layers.
        """

        explorer = NuScenesMapExplorer(self.nusc_map) #TODO: initialize in init()?

        # Settings
        patch_margin = 2
        min_diff_patch = 30

        # Ids of scenes with a bad match between localization and map.
        #scene_blacklist = [499, 515, 517]

        if verbose:
            print('Creating plot...')
        map_poses = np.vstack(map_poses)[:, :2]

        # Render the map patch with the current ego poses.
        min_patch = np.floor(map_poses.min(axis=0) - patch_margin)
        max_patch = np.ceil(map_poses.max(axis=0) + patch_margin)
        diff_patch = max_patch - min_patch
        if any(diff_patch < min_diff_patch):
            center_patch = (min_patch + max_patch) / 2
            diff_patch = np.maximum(diff_patch, min_diff_patch)
            min_patch = center_patch - diff_patch / 2
            max_patch = center_patch + diff_patch / 2
        my_patch = (min_patch[0], min_patch[1], max_patch[0], max_patch[1])
        fig, ax = explorer.render_map_patch(my_patch, explorer.map_api.non_geometric_layers, figsize=(10, 10),
                                        render_egoposes_range=render_egoposes_range,
                                        render_legend=render_legend)

        # Plot in the same axis as the map.
        # Make sure these are plotted "on top".
        if render_egoposes:
            ax.scatter(map_poses[:, 0], map_poses[:, 1], s=20, c='k', alpha=1.0, zorder=2)
        plt.axis('off')
        
        if out_path is not None:
            plt.savefig(f'/example/renderings/ego_poses_{datetime.now()}.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        #return map_poses, fig, ax


    def render_rectangle_agent_scene(self, x,y,yaw, agent_size=(2,4)):
            #render agent as a rectangle and show its heading direction vector compared to the direction of the lane, for each sample in the scene.
            road_segment_token = self.nusc_map.record_on_point(x,y, 'road_segment')
            current_lane = self.nusc_map.record_on_point(x,y, 'lane')

            if road_segment_token and self.nusc_map.get('road_segment', road_segment_token)['is_intersection'] and not current_lane:
                closest_lane = self.nusc_map.get_closest_lane(x, y, radius=2)
                lane_path = self.nusc_map.get_arcline_path(closest_lane)
                closest_pose_idx_to_lane, lane_record, _ = SelfDrivingEnvironment.project_pose_to_lane((x, y, yaw), lane_path)
                if closest_pose_idx_to_lane == len(lane_record) - 1:
                    tangent_vector = lane_record[closest_pose_idx_to_lane] - lane_record[closest_pose_idx_to_lane - 1]
                else:
                    tangent_vector = lane_record[closest_pose_idx_to_lane + 1] - lane_record[closest_pose_idx_to_lane]

            else:

                lane = self.nusc_map.get_arcline_path(current_lane)
                closest_pose_idx_to_lane, lane_record, distance_along_lane = SelfDrivingEnvironment.project_pose_to_lane((x, y, yaw), lane)
                if closest_pose_idx_to_lane == len(lane_record) - 1:
                        tangent_vector = lane_record[closest_pose_idx_to_lane] - lane_record[closest_pose_idx_to_lane - 1]
                else:
                    tangent_vector = lane_record[closest_pose_idx_to_lane + 1] - lane_record[closest_pose_idx_to_lane]

    
              
            patch_size = 20
            patch_box = [x,y, patch_size, patch_size]
            patch = NuScenesMapExplorer.get_patch_coord(patch_box)
            minx, miny, maxx, maxy = patch.bounds

            fig, ax = self.nusc_map.render_map_patch( [minx, miny, maxx, maxy], ['road_divider', 'lane_divider'], figsize=(5, 5))
            
            ax.scatter(x,y)
            heading_vector = np.array([np.cos(yaw), np.sin(yaw)])

            yaw =  math.degrees(-(math.pi / 2) + yaw)
            rotated_rectangle = create_rectangle((x,y), yaw, agent_size)
            ax.quiver(x, y, heading_vector[0], heading_vector[1], color='b', scale=10, label='Ego Direction')
            ax.quiver(x,y, tangent_vector[0], tangent_vector[1],  color='r', scale=10, label='Lane Direction')
            x,y = rotated_rectangle.exterior.xy
            ax.plot(x,y)

    
    
    
    ########################
    ## PROCESS ENVIRONEMNT 
    ########################

    '''

    def keep_drivable_area(self,x,y):
        is_lane_area = self.nusc_map.record_on_point(x, y, 'lane') 
        is_road = self.nusc_map.record_on_point(x, y, 'road_segment')
        #TODO: exclude road_blocks from road_segment
        if len(is_lane_area) == 0 and len(is_road)==0:
            x,y = self.reach_drivable_area(x,y)
        
        return x,y

    '''


    def reach_drivable_area(self, x,y, radius:float=5, resolution_meters:float = 0.5):
        """
        Get closest lane id within a radius of query point. The distance from a point (x, y) to a lane is
        the minimum l2 distance from (x, y) to a point on the lane.
        Then, find the closest pose on this lane.
        Note that this function does not take the heading of the query pose into account.
        
        Args:
            x: X coordinate in global coordinate frame.
            y: Y Coordinate in global coordinate frame.
            radius: Radius around point to consider.        
            resolution_meters:How finely to discretize the lane.
        Return: 
            Tuple of the closest pose along the lane
        """

        lanes = self.nusc_map.get_records_in_radius(x, y, radius, ['lane', 'lane_connector'])
        lanes = lanes['lane'] + lanes['lane_connector']
        discrete_points = self.nusc_map.discretize_lanes(lanes, resolution_meters) #returns ID, points for each lane

        current_min = np.inf
        closest_pose = (None, None)
        
        for lane_id, points in discrete_points.items():
            points_array = np.array(points)
            distances = np.linalg.norm(points_array[:, :2] - [x, y], axis=1)
            min_distance = distances.min()
            if min_distance <=current_min:
                current_min = min_distance
                closest_pose_index = distances.argmin()
                closest_pose = (points_array[closest_pose_index, 0], points_array[closest_pose_index, 1])

        return closest_pose
    


    
   


    ###################################
    #PROCESS NEARBY OBJECTS INFORMATION
    ###################################

    def is_near_stop_sign(self, x,y, front_area:Polygon):
        """
        Check if there is a stop sign or yield nearby the given pose (x, y).

        Args:
            x (float): Current x-coordinate of the vehicle.
            y (float): Current y-coordinate of the vehicle.
            front_area (Polygon): Area in front of the vehicle to check for signs.

        Returns:
            bool: True if a sign is nearby the ego car in the same road block or at an interesection, False otherwise.
        
        """        

        current_road_block = self.nusc_map.record_on_point(x,y, 'road_block')

        for stop_line in self.nusc_map.stop_line:
            if stop_line['stop_line_type'] in ['STOP_SIGN','YIELD']:
                    stop_line_polygon = self.nusc_map.extract_polygon(stop_line['polygon_token'])                
                    if stop_line_polygon.intersects(front_area):
                        #print(f'A sign is intersecting: {stop_line}')

                        if stop_line['road_block_token'] == current_road_block or current_road_block == '': #or intersection
                                return stop_line['stop_line_type']
                
        return None



    def is_near_ped_crossing(self,front_area:Polygon):
        """
        Check if there is a pedestrian crossing or a turn stop with a pedestrian crossing nearby.

        Args:
            front_area (Polygon): Area in front of the vehicle to check for pedestrian crossings.

        Returns:
            bool: True if a pedestrian crossing or a turn stop with a pedestrian crossing is nearby, False otherwise.
        """
            
        for ped_crossing in self.nusc_map.ped_crossing:  
            ped_crossing_polygon = self.nusc_map.extract_polygon(ped_crossing['polygon_token'])
            if ped_crossing_polygon.intersects(front_area):
                #print('Zebra crossing nearby')
                return True

        for stop_line in self.nusc_map.stop_line:
            if stop_line['stop_line_type'] == 'TURN_STOP':
                    stop_line_polygon = self.nusc_map.extract_polygon(stop_line['polygon_token'])
                    if stop_line_polygon.intersects(front_area):
                        if stop_line['ped_crossing_tokens']:
                            #print('Zebra crossing neaby (turn stop)')
                            return True 
                        else:
                            print('Turn stop without zebra crossing. To be handled.)') #TODO:
            return False

    
    def is_near_traffic_light(self, yaw, front_area:Polygon, eps=0.1):
        
        """
        Check if there is a traffic light nearby the given pose (x, y).

        Args:
            yaw (float): Yaw angle of the vehicle in radians.
            front_area (Polygon): Area for detecting traffic light.
            eps (float): Epsilon value for alignment tolerance.

        Returns:
            bool: True if a traffic light is nearby and aligned with the vehicle's direction of travel, False otherwise.
        """

        for traffic_light in self.nusc_map.traffic_light:
                line = self.nusc_map.extract_line(traffic_light['line_token'])
                xs, ys = line.xy
                point = Point(xs[0], ys[0])# Traffic light is represented as a line, we take the starting point
                if point.within(front_area):
                    traffic_light_direction = (xs[1] - xs[0], ys[1] - ys[0])   
                    alignment = determine_travel_alignment(traffic_light_direction, yaw)
                    #print(f'traffic light alignmenet {alignment}')
                    if alignment <-eps:
                         #print('valid traffic light')
                        return True
        
        for stop_line in self.nusc_map.stop_line:
            if stop_line['stop_line_type'] == 'TRAFFIC_LIGHT' and stop_line['traffic_light_tokens']:
                    stop_line_polygon = self.nusc_map.extract_polygon(stop_line['polygon_token'])
                    if stop_line_polygon.intersects(front_area):
                            for token in stop_line['traffic_light_tokens']:
                                traffic_light = self.nusc_map.get('traffic_light', token)
                                line = self.nusc_map.extract_line(traffic_light['line_token'])
                                xs, ys = line.xy                              
                                traffic_light_direction = (xs[1] - xs[0], ys[1] - ys[0])   
                                alignment = determine_travel_alignment(traffic_light_direction, yaw)
                                #print(f'traffic light alignmenet {alignment}')
                                if alignment <-eps:
                                    #print('valid traffic light stop line')
                                    return True

        return False



    


    #########################
    #PROCESS LANE INFORMATION
    #########################

      
    def is_on_divider(self, x,y, yaw, agent_size:Tuple[float, float]) -> bool:
        """
        Checks whether the ego vehicle interescts lane and road dividers
        :param x,y,yaw: coordinates (in meters) and heading (in radians) of the agent 
        :param agent_size: height and width of the box representing the agent
        :return: True if agent intersects the layers specified in layer_name
        """
        yaw =  math.degrees(-(math.pi / 2) + yaw)
        
        # rectangle centered at (x, y) and rotated of yaw angle representing the agent
        rotated_rectangle = create_rectangle((x,y), yaw, agent_size)

        for record in self.dividers:
            line = self.nusc_map.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = rotated_rectangle.intersection(line)
            if not new_line.is_empty:
                return True

        return False  

    @staticmethod
    def project_pose_to_lane(pose, lane: List[arcline_path_utils.ArcLinePath], resolution_meters: float = 1):
        """
        Find the closest pose on a lane to a query pose and additionally return the
        distance along the lane for this pose. Note that this function does
        not take the heading of the query pose into account.
        :param pose: Query pose in (x,y) coordinates.
        :param lane: Will find the closest pose on this lane.
        :param resolution_meters: How finely to discretize the lane.
        :return: Tuple of the closest pose index, discretized xy points of the line and distance along the lane
        """

        discretized_lane = arcline_path_utils.discretize_lane(lane, resolution_meters=resolution_meters)

        xy_points = np.array(discretized_lane)[:, :2]
        closest_pose_index = np.linalg.norm(xy_points - pose[:2], axis=1).argmin()
        distance_along_lane = closest_pose_index *resolution_meters
        return closest_pose_index, xy_points, distance_along_lane
    



    def get_position_predicates(self,x,y, yaw, eps=0.3, agent_size:Tuple[float, float]=(2,4)):
        """
        Determines the lane progress (which chunk the distance along the lane falls into). The lane is divided into 3 equal chunks.
        Determines the lane position (Left if on the left lane of the road, RIGHT is right and Center if in the center.)
        
             
        :return: (BlockProgress, LanePosition)
        """
        #NOTE: Lanes are not always straight. The approach is such to account for eventual curvature. 
        #print('Road objects on selected point:', nusc_map.layers_on_point(x, y), '\n')

        drivable_area = self.nusc_map.record_on_point(x,y, 'drivable_area')
        if not drivable_area:
            return (BlockProgress.NONE, LanePosition.NONE)
        
        road_segment_token = self.nusc_map.record_on_point(x,y, 'road_segment')
        current_lane = self.nusc_map.record_on_point(x,y, 'lane')

        if road_segment_token and self.nusc_map.get('road_segment', road_segment_token)['is_intersection'] and not current_lane:
            
            if self.is_on_divider(x,y, yaw, agent_size):
                return (BlockProgress.INTERSECTION, LanePosition.CENTER)
            else:
                closest_lane = self.nusc_map.get_closest_lane(x, y, radius=2)
                lane_path = self.nusc_map.get_arcline_path(closest_lane)
                closest_pose_idx_to_lane, lane_record, _ = SelfDrivingEnvironment.project_pose_to_lane((x, y, yaw), lane_path)

                #if closest_pose_idx_to_lane == len(lane_record) - 1:
                #    direction_vector = lane_record[closest_pose_idx_to_lane] - lane_record[closest_pose_idx_to_lane - 1]
                #else:
                #    direction_vector = lane_record[closest_pose_idx_to_lane + 1] - lane_record[closest_pose_idx_to_lane]

                if closest_pose_idx_to_lane == len(lane_record) - 1:
                    direction_vector = lane_record[closest_pose_idx_to_lane] - lane_record[closest_pose_idx_to_lane - 2]
                
                elif closest_pose_idx_to_lane== 0:
                    direction_vector = lane_record[closest_pose_idx_to_lane + 2] - lane_record[closest_pose_idx_to_lane]
                else:
                    direction_vector = lane_record[closest_pose_idx_to_lane + 1] - lane_record[closest_pose_idx_to_lane-1]

   

                direction_of_travel = determine_travel_alignment(direction_vector, yaw)

                if direction_of_travel <-eps:
                    return(BlockProgress.INTERSECTION, LanePosition.LEFT)
                elif direction_of_travel>eps:
                    return(BlockProgress.INTERSECTION,LanePosition.RIGHT)
                else:
                    return(BlockProgress.INTERSECTION,LanePosition.NONE) 
                    
        block_progress = None
        lane_position = None
        
        if not current_lane:
            current_lane = self.nusc_map.get_closest_lane(x, y, radius=2)                   
        lane = self.nusc_map.get_arcline_path(current_lane)
        closest_pose_idx_to_lane, lane_record, distance_along_lane = SelfDrivingEnvironment.project_pose_to_lane((x, y, yaw), lane)

        #1. Determine Block Progress
        chunk_size = arcline_path_utils.length_of_lane(lane) / 3

        if distance_along_lane < chunk_size:
             block_progress = BlockProgress.START 
        elif distance_along_lane < 2*chunk_size:
            block_progress = BlockProgress.MIDDLE
        else: 
            block_progress =  BlockProgress.END 



        #2. Determine Lane Position
        if self.is_on_divider(x,y, yaw, agent_size):
            lane_position = LanePosition.CENTER
        else:
            #if closest_pose_idx_to_lane == len(lane_record) - 1:
            #    direction_vector = lane_record[closest_pose_idx_to_lane] - lane_record[closest_pose_idx_to_lane - 1]
            #else:
            #    direction_vector = lane_record[closest_pose_idx_to_lane + 1] - lane_record[closest_pose_idx_to_lane]

            if closest_pose_idx_to_lane == len(lane_record) - 1:
                direction_vector = lane_record[closest_pose_idx_to_lane] - lane_record[closest_pose_idx_to_lane - 2]
            
            elif closest_pose_idx_to_lane== 0:
                direction_vector = lane_record[closest_pose_idx_to_lane + 2] - lane_record[closest_pose_idx_to_lane]
            else:
                direction_vector = lane_record[closest_pose_idx_to_lane + 1] - lane_record[closest_pose_idx_to_lane-1]


            direction_of_travel = determine_travel_alignment(direction_vector, yaw)
                    
            if direction_of_travel <-eps:
                lane_position = LanePosition.LEFT
            elif direction_of_travel>eps:
                lane_position = LanePosition.RIGHT
            else: 
                lane_position = LanePosition.NONE 
                #NOTE: direction vector close to zero, meaning that 1) 2 points in the arcline path are very close or 2) that the closest lane to the vehicle at 
                # an intersection has direction perpendicular to the one of the ego car.
                #fix for 1): use prev or next points.
        
        return (block_progress, lane_position)
    


    def get_lane_position(self, x, y, yaw, eps=0.3, agent_size: Tuple[float, float] = (2, 4)):
        """
        Determines the lane position (Left if on the left lane of the road, RIGHT if on the right lane, and Center if in the center.)
        
        :return: LanePosition
        """
        
        if not self.nusc_map.record_on_point(x, y, 'drivable_area'):
            return LanePosition.NONE

        if self.is_on_divider(x, y, yaw, agent_size):
            return LanePosition.CENTER
        
        lane_position = None
        current_lane = self.nusc_map.record_on_point(x, y, 'lane')
        if not current_lane:
            current_lane = self.nusc_map.get_closest_lane(x, y, radius=2)
        lane = self.nusc_map.get_arcline_path(current_lane)
        closest_pose_idx_to_lane, lane_record, _ = SelfDrivingEnvironment.project_pose_to_lane((x, y, yaw), lane)

        if closest_pose_idx_to_lane == len(lane_record) - 1:
            direction_vector = lane_record[closest_pose_idx_to_lane] - lane_record[closest_pose_idx_to_lane - 2]
        elif closest_pose_idx_to_lane == 0:
            direction_vector = lane_record[closest_pose_idx_to_lane + 2] - lane_record[closest_pose_idx_to_lane]
        else:
            direction_vector = lane_record[closest_pose_idx_to_lane + 1] - lane_record[closest_pose_idx_to_lane - 1]

        direction_of_travel = determine_travel_alignment(direction_vector, yaw)

        if direction_of_travel < -eps:
            lane_position = LanePosition.LEFT
        elif direction_of_travel > eps:
            lane_position = LanePosition.RIGHT
        else:
            lane_position = LanePosition.NONE

        return lane_position
            

    ##############################
    # PEDESTRIANS AND TWO-WHEELERS
    ##############################          
    @staticmethod
    def is_vulnerable_subject_nearby(state_detections):

        """
        Function to check for vulnerable subjects nearby based on state detections from all cameras.
        Vulnerable subjects include pedestrians, cyclists and people driving scooters.
        
        Parameters:
            state_detections (dict): A dictionary where keys are camera types and values are 
                                    serialized object detection results in string format at a given state.
        
        Returns:
            tuple: A tuple containing the total count of pedestrians and cyclists.
        
        NOTE: Only subjects with >40% visibility are considered.
        """
        tot_ped_count = 0
        tot_bike_count = 0
        for cam_type, objects in state_detections.items():

                for (category, attribute), count in ast.literal_eval(objects).items():
                    if 'human.pedestrian' in category:
                        if 'personal_mobility' in category and 'with_rider' in attribute: #scooter
                            tot_bike_count+=count
                        else:
                            tot_ped_count +=count
                    elif 'cycle' in category and 'with_rider' in attribute:
                        tot_bike_count +=count

                return tot_ped_count, tot_bike_count