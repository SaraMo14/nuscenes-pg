import argparse
from pathlib import Path
import pandas as pd
from nuscenes.utils.geometry_utils import BoxVisibility

import json
import time
from pyquaternion import Quaternion
from typing import List
import numpy as np
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import box_in_image
from table_loader import BaseTableLoader
import matplotlib.pyplot as plt
import os.path as osp
from PIL import Image


class CamDataProcessor(BaseTableLoader):
    
    def __init__(self, dataroot, dataoutput, version, alpha):#, complexity):
        super().__init__(dataroot, version)
        
        self.dataroot = dataroot
        self.dataoutput = dataoutput
        self.version = version
        self.alpha = alpha
        #self.complexity = complexity
        
        self.cameras = ['CAM_FRONT']
        
        self.table_names = ['attribute','ego_pose','sensor', 'category', 'visibility','calibrated_sensor','sample_data', 'instance','sample', 'sample_annotation']

      
        start_time = time.time()
        print("======\nLoading NuScenes tables for version {}...".format(self.version))

        self.sample_data = self.__load_table__('sample_data', drop_fields=['next', 'prev', 'timestamp','fileformat'])
        print('sample_data loaded')
        self.ego_pose = self.__load_table__('ego_pose', drop_fields=['timestamp'])     
        print('ego_pose loaded')
        self.calibrated_sensor = self.__load_table__('calibrated_sensor')
        print('calibrated_sensor loaded')
        self.sensor = self.__load_table__('sensor')
        print('sensor loaded')
        self.attribute = self.__load_table__('attribute', drop_fields=['description'])
        print('attribute loaded')
        self.instance = self.__load_table__('instance')
        print('instance loaded')
        self.visibility = self.__load_table__('visibility', drop_fields=['description'])
        print('visibility loaded')
        self.category = self.__load_table__('category', drop_fields=['description', 'index'])
        print('category loaded')
        self.sample_annotation = self.__load_table__('sample_annotation', drop_fields=['prev','next','num_lidar_pts', 'num_radar_pts'])
        print('sample_annotation loaded')
        
        self.sample = self.__load_table__('sample')
        print('sample loaded')
        self.__make_reverse_index__(verbose=True)
        print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))

   
    def cam_detection(self, sample_tokens:pd.DataFrame):
        """
        Given a sample in a scene, returns the objects in front of the vehicle, the action they are performing,
        and the visibility (0=min, 4=100%) from the ego-vehicle.

        NOTE:
        A sample of a scene (frame) has several sample annotations (Bounding Boxes). Each sample annotation
        has 0, 1, or + attributes (e.g., pedestrian moving, etc).
        The instance of an annotation is described in the instance table, which tracks the number of annotations
        in which the object appears.

        For each sample, check if there are any annotations. Retrieve the list of annotations for the sample.
        For each annotation, check from which camera it is from.

        """
        for sample_token in sample_tokens['token']:
            sample = self.get('sample', sample_token)
            sample_detected_objects = {cam_type: {} for cam_type in self.cameras}
            
            if sample.get('anns'): #if sample has annotated objects
                #####à
                #fig, ax = plt.subplots(1, 1, figsize=(9, 9))
                #sd_record = self.get('sample_data', sample['data']['CAM_FRONT'])
                #data_path = osp.join(self.dataroot, sd_record['filename'])
                #im = Image.open(data_path)
                #ax.imshow(im)
                ########


                for ann_token in sample['anns']:
                    ann_info = self.get('sample_annotation', ann_token)
                    visibility = int(self.get('visibility', ann_info['visibility_token'])['token'])
                    if visibility >=3:
                        category = ann_info['category_name']
                        for cam_type in self.cameras:
                            box = self.get_sample_data(
                                sample['data'][cam_type], 
                                box_vis_level=BoxVisibility.ANY,
                                selected_anntoken=ann_token
                            )
                            if box:
                                #filter objects within a distance alpha.
                                #box.render(ax, view= np.array(self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])['camera_intrinsic']), normalize=True, colors=('r','r','r'))
                                        
                                if ann_info['attribute_tokens']:#only vehicles, cyclists and pedestrians have attributes
                                    for attribute in ann_info['attribute_tokens']: #an annotation can have more than one attribute in time (e.g. pedestrian moving/not moving)
                                        attribute_name = self.get('attribute', attribute)['name']
                                        key = (category, attribute_name)
                                        if key not in sample_detected_objects[cam_type]:
                                            sample_detected_objects[cam_type][key] = 0
                                        sample_detected_objects[cam_type][key] += 1
                                else: #movable objects, static objects etc
                                    key = (category, '') #no attribute name
                                    if key not in sample_detected_objects[cam_type]:
                                        sample_detected_objects[cam_type][key] = 0
                                    sample_detected_objects[cam_type][key] += 1
                
                #ax.axis('off')
                #if sample_token == '31812a5e8d514b5f8d2fbc50fc007475':
                    #plt.title(sample_token)
                    #plt.tight_layout()
                    #plt.savefig(f'{sample_token}_{self.alpha}.png')
               
            for cam_type in self.cameras:
                sample_tokens.loc[sample_tokens['token'] == sample_token, f'{cam_type}'] = str(sample_detected_objects[cam_type])

        
        df_detected_objects = pd.DataFrame(sample_tokens).rename(columns={'token': 'sample_token'})#, columns=['sample_token', 'detected_objects'])
        output_path = Path(self.dataoutput) / 'cam_detection.csv'
        df_detected_objects.to_csv(output_path, index=False)

        print(f"Camera detection data saved to {output_path}")
        

    def get(self, table_name: str, token: str) -> dict:
        """
        Returns a record from table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: Table record. See README.md for record details for each table.
        """
        assert table_name in self.table_names, "Table {} not found".format(table_name)

        return getattr(self, table_name)[self.getind(table_name, token)]
    
    def getind(self, table_name: str, token: str) -> int:
        """
        This returns the index of the record in a table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: The index of the record in table, table is an array.
        """
        return self._token2ind[table_name][token]
    
    def __make_reverse_index__(self, verbose: bool) -> None:
        """
        De-normalizes database to create reverse indices for common cases.
        :param verbose: Whether to print outputs.
        """

        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.
        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()

            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member['token']] = ind

        # Decorate (adds short-cut) sample_annotation table with for category name.
        for record in self.sample_annotation:
            inst = self.get('instance', record['instance_token'])
            record['category_name'] = self.get('category', inst['category_token'])['name']

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.sample_data:
            cs_record = self.get('calibrated_sensor', record['calibrated_sensor_token'])
            sensor_record = self.get('sensor', cs_record['sensor_token'])
            record['sensor_modality'] = sensor_record['modality']
            record['channel'] = sensor_record['channel']

        # Reverse-index samples with sample_data and annotations.
        for record in self.sample:
            record['data'] = {}
            record['anns'] = []

        for record in self.sample_data:
            if record['is_key_frame']:
                sample_record = self.get('sample', record['sample_token'])
                sample_record['data'][record['channel']] = record['token']

        for ann_record in self.sample_annotation:
            sample_record = self.get('sample', ann_record['sample_token'])
            sample_record['anns'].append(ann_record['token'])


        if verbose:
            print("Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time))

    
    def get_sample_data(self, sample_data_token: str,
                        box_vis_level: BoxVisibility = BoxVisibility.ANY,
                        selected_anntoken: str = None,
                        use_flat_vehicle_coordinates: bool = False) -> Box:
        """
        Returns the data path as well as all annotations related to that sample_data.
        Note that the boxes are transformed into the current sensor's coordinate frame.
        :param sample_data_token: Sample_data token.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param selected_anntoken: If provided only return the selected annotation.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                             aligned to z-plane in the world.
        :param radius: if specified, the funcion returns all boxes that are far at most <alpha> from the ego vehicle.
        :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
        """

        # Retrieve sensor & pose records
        sd_record = self.get('sample_data', sample_data_token)
        cs_record = self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = self.get('sensor', cs_record['sensor_token'])
        pose_record = self.get('ego_pose', sd_record['ego_pose_token'])

        #data_path = self.get_sample_data_path(sample_data_token)

        if sensor_record['modality'] == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])
        else:
            cam_intrinsic = None
            imsize = None

        box = self.get_box(selected_anntoken)
        if self.alpha is None or self.distance_ego_to_object(pose_record['translation'], box) < self.alpha:
            if use_flat_vehicle_coordinates:
                # Move box to ego vehicle coord system parallel to world z plane.
                yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            else:
                # Move box to ego vehicle coord system.
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)
                #  Move box to sensor coord system.
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

            if sensor_record['modality'] == 'camera' and not box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
                return None
                
            return  box
        return None


    def get_box(self, sample_annotation_token: str) -> Box:
        """
        Instantiates a Box class from a sample annotation record.
        :param sample_annotation_token: Unique sample_annotation identifier.
        """
        record = self.get('sample_annotation', sample_annotation_token)
        return Box(record['translation'], record['size'], Quaternion(record['rotation']),
                   name=record['category_name'], token=record['token'])

    @staticmethod
    def distance_ego_to_object(ego_position, box):
        """
        Compute the distance from the ego vehicle to the closest part of the detected object.

        :param ego_position: A tuple (x, y, z) representing the position of the ego vehicle.
        :param box: A Box object representing the detected object.
        :return: The minimum distance from the ego vehicle to the closest point on the object.
        
        """
        corners = box.bottom_corners()
        distances = np.linalg.norm(corners.T - np.array(ego_position), axis = 1)
        return np.min(distances)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process nuScenes camera data and save to a file.")
    parser.add_argument('--dataroot', required=True, type=str, help='Path to the nuScenes dataset directory.')
    parser.add_argument('--dataoutput', required=True, type=str, help='Path for the output data file directory.')
    parser.add_argument('--version', required=True, type=str, choices=["v1.0-mini", "v1.0-trainval"], help='Version of the nuScenes dataset to process.')
    parser.add_argument('--alpha', type=float, default=None, help='Distance within consider detections. Defaults to None if not specified.')
    #parser.add_argument('--complexity', required=True, type=int, default=0, choices=[0,1], help='Level of complexity of the dataset.')

    args = parser.parse_args()
    processor = CamDataProcessor(args.dataroot, args.dataoutput, args.version, args.alpha)#,args.complexity)
    
   #sample_tokens = pd.DataFrame(processor.sample)[pd.DataFrame(processor.sample)['scene_token'] == 'bebf5f5b2a674631ab5c88fd1aa9e87a']['token'].to_frame()
        
    sample_tokens = pd.DataFrame(processor.sample)['token'].to_frame()

    processor.cam_detection(sample_tokens)

#python3 create_cam_data.py --dataroot /home/saramontese/Desktop/MasterThesis/example/dataset/data/sets/nuscenes --dataoutput /home/saramontese/Desktop/MasterThesis/example/dataset/data/sets/nuscenes --version v1.0-trainval --alpha 10

