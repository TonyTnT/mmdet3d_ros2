import os
import ctypes
import struct
import time

import numpy as np
import torch
import mmcv

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import ReliabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, TransformStamped
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
import tf2_ros
import tf_transformations

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.models.layers import aligned_3d_nms
from mmdet3d.structures import DepthInstance3DBoxes

def transform_point(trans, pt):
    # https://answers.ros.org/question/249433/tf2_ros-buffer-transform-pointstamped/
    quat = [
        trans.transform.rotation.x,
        trans.transform.rotation.y,
        trans.transform.rotation.z,
        trans.transform.rotation.w
    ]
    mat = tf_transformations.quaternion_matrix(quat)
    pt_np = [pt.x, pt.y, pt.z, 1.0]
    pt_in_map_np = np.dot(mat, pt_np)

    pt_in_map = Point()
    pt_in_map.x = pt_in_map_np[0] + trans.transform.translation.x
    pt_in_map.y = pt_in_map_np[1] + trans.transform.translation.y
    pt_in_map.z = pt_in_map_np[2] + trans.transform.translation.z

    return pt_in_map

class InferNode(Node):
    def __init__(self):
        super().__init__('infer_node')
        self.logger = self.get_logger()

        cache_time = Duration(seconds=2.0) 
        self.tf_buffer = tf2_ros.Buffer(cache_time)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.declare_parameter('config_file', 'projects/TR3D/configs/tr3d_1xb16_sunrgbd-3d-10class.py')
        self.declare_parameter('checkpoint_file', '../checkpoints/tr3d_1xb16_sunrgbd-3d-10class.pth')
        self.declare_parameter('point_cloud_frame', 'femto_mega_color_optical_frame')
        self.declare_parameter('point_cloud_topic', '/femto_mega/depth_registered/filter_points')
        self.declare_parameter('score_threshold', 0.98)
        self.declare_parameter('infer_device', 'cuda:0')
        self.declare_parameter('nms_interval', 0.5)
        self.declare_parameter('point_cloud_qos', 'best_effort')
        # votenet
        # self.declare_parameter('config_file', 'configs/votenet/votenet_8xb16_sunrgbd-3d.py')
        # self.declare_parameter('checkpoint_file', '../checkpoints/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth')
        # imvoxelnet
        # self.declare_parameter('config_file', 'configs/imvoxelnet/imvoxelnet_2xb4_sunrgbd-3d-10class.py')
        # self.declare_parameter('checkpoint_file', '../checkpoints/imvoxelnet_4x2_sunrgbd-3d-10class_20220809_184416-29ca7d2e.pth')

        config_file_path = self.get_parameter('config_file').get_parameter_value().string_value
        checkpoint_file_path = self.get_parameter('checkpoint_file').get_parameter_value().string_value
        infer_device = self.get_parameter('infer_device').get_parameter_value().string_value
        self.score_thrs = self.get_parameter('score_threshold').get_parameter_value().double_value
        nms_interval = self.get_parameter('nms_interval').get_parameter_value().double_value
        self.point_cloud_frame = self.get_parameter('point_cloud_frame').get_parameter_value().string_value
        point_cloud_qos = self.get_parameter('point_cloud_qos').get_parameter_value().string_value
        point_cloud_topic = self.get_parameter('point_cloud_topic').get_parameter_value().string_value

        qos = QoSProfile(depth=5)
        if point_cloud_qos == 'best_effort':
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
        elif point_cloud_qos == 'reliable':
            qos.reliability = ReliabilityPolicy.RELIABLE
        else:
            self.logger.error('Invalid value for point_cloud_qos parameter')
            return

        self.transform_stamped = TransformStamped()
        self.det3d_array = Detection3DArray()
        self.det3d_array.header.frame_id = 'odom'

        if 'sunrgbd' in checkpoint_file_path:
            self.class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
                   'night_stand', 'bookshelf', 'bathtub')
        elif 'scannet' in checkpoint_file_path:
            self.class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                   'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                   'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                   'garbagebin')
        else:
            self.logger.error('Unknown weight, path of weight should contain "sunrgbd" or "scannet"')

        self.get_logger().info('full_config_file: "%s"' % config_file_path)
        self.get_logger().info('checkpoint_file: "%s"' % checkpoint_file_path)
        self.model = init_model(config_file_path, checkpoint_file_path, device=infer_device)

        self.subscription = self.create_subscription(
            PointCloud2,
            point_cloud_topic,
            self.listener_callback,
            qos)
        self.marker_pub = self.create_publisher(Detection3DArray, '/detect_bbox3d', 10)
        # for debug
        # self.publisher_ = self.create_publisher(PointCloud2, '/detect_bbox_infer_pcd', qos)
        # for nms and publish
        self.timer = self.create_timer(nms_interval, self.detections_callback)

        self.filtered_bboxes_nms = torch.zeros(0, 6).cuda()
        self.filtered_bboxes_tensor = torch.zeros(0, 7).cuda()
        self.filtered_scores = torch.zeros(0).cuda()
        self.filtered_labels = torch.zeros(0).cuda()


    def listener_callback(self, msg):
        # convert pointcloud2 to points with color (xyzrgb)
        gen = pc2.read_points(msg, skip_nans=True)
        int_data = list(gen)
        # Initialize an array with the correct size        
        color_points = np.zeros((len(int_data), 6))
        # points 是相机坐标系下的
        points = np.zeros((len(int_data), 3))
        # base points 是odom 坐标系下的
        base_points = np.zeros((len(int_data), 3))
        try:
            transform_stamped = self.tf_buffer.lookup_transform('odom', self.point_cloud_frame, msg.header.stamp)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().error('Failed to lookup transform')
            return
        for ind, x in enumerate(int_data):
            test = x[3] 
            points[ind] = [x[0], x[1], x[2]]
            pt = Point()
            pt.x, pt.y, pt.z = x[0], x[1], x[2]
            base_pt = transform_point(transform_stamped, pt)
            s = struct.pack('>f' ,test)
            i = struct.unpack('>l',s)[0]
            pack = ctypes.c_uint32(i).value
            r = ((pack & 0x00FF0000)>> 16) /255
            g = ((pack & 0x0000FF00)>> 8) /255
            b = (pack & 0x000000FF) / 255
            # Fill the array
            color_points[ind] = [base_pt.x, base_pt.y, base_pt.z, r, g, b]
            base_points[ind] = [base_pt.x, base_pt.y, base_pt.z]
        
        # infer_points = pc2.create_cloud_xyz32(header=msg.header, points=base_points)
        # self.publisher_.publish(infer_points)
        start_time = time.time()  # get current time
        # for sunrgbd
        model_result, data_afterprocess = inference_detector(self.model, color_points)
        end_time = time.time()  # get current time after inference
        # Calculate elapsed time in milliseconds
        elapsed_time_ms = (end_time - start_time) * 1000
        self.logger.debug("Inference time: {:.2f} ms".format(elapsed_time_ms))

        bboxes = model_result.pred_instances_3d.bboxes_3d
        scores = model_result.pred_instances_3d.scores_3d
        labels = model_result.pred_instances_3d.labels_3d
        indices = torch.where(scores > self.score_thrs)
        # x_center, y_center, z_center, dx, dy, dz, yaw
        filtered_bboxes = bboxes[indices]
        filtered_scores = scores[indices]
        filtered_labels = labels[indices]
        
        if filtered_bboxes.shape[0] != 0:
            filtered_bboxes_x0 = filtered_bboxes.center[:,0]-0.5*filtered_bboxes.dims[:,0]
            filtered_bboxes_y0 = filtered_bboxes.center[:,1]-0.5*filtered_bboxes.dims[:,1]
            filtered_bboxes_z0 = filtered_bboxes.center[:,2]-0.5*filtered_bboxes.dims[:,2]
            filtered_bboxes_x1 = filtered_bboxes.center[:,0]+0.5*filtered_bboxes.dims[:,0]
            filtered_bboxes_y1 = filtered_bboxes.center[:,1]+0.5*filtered_bboxes.dims[:,1]
            filtered_bboxes_z1 = filtered_bboxes.center[:,2]+0.5*filtered_bboxes.dims[:,2]
            filtered_bboxes_nms = torch.stack((filtered_bboxes_x0, filtered_bboxes_y0,
                                               filtered_bboxes_z0, filtered_bboxes_x1,
                                               filtered_bboxes_y1, filtered_bboxes_z1), dim=1)
            self.filtered_bboxes_nms = torch.cat((self.filtered_bboxes_nms, filtered_bboxes_nms), dim=0)
            self.filtered_bboxes_tensor = torch.cat((self.filtered_bboxes_tensor, filtered_bboxes.tensor), dim=0)
            self.filtered_scores = torch.cat((self.filtered_scores, filtered_scores), dim=0)
            self.filtered_labels = torch.cat((self.filtered_labels, filtered_labels), dim=0)

    def detections_callback(self):
        pick_ind = aligned_3d_nms(self.filtered_bboxes_nms, self.filtered_scores, self.filtered_labels, 0.25)
        self.logger.info("[NMS] detections {} -> {}".format(self.filtered_bboxes_nms.shape[0], pick_ind.shape[0]))
        self.filtered_bboxes_nms = self.filtered_bboxes_nms[pick_ind]
        self.filtered_labels = self.filtered_labels[pick_ind]
        self.filtered_scores = self.filtered_scores[pick_ind]
        self.filtered_bboxes_tensor = self.filtered_bboxes_tensor[pick_ind]
        self.draw_bbox(self.filtered_bboxes_tensor.cpu(), self.filtered_labels.cpu().numpy(), self.filtered_scores.cpu().numpy())
    
    def draw_bbox(self, bboxes, labels, scores, timestamp=None):
        det3d_array = Detection3DArray()
        det3d_array.header.frame_id = 'odom'
        if len(bboxes) > 0:
            for ind in range(len(bboxes)):
                bbox = bboxes[ind]
                label = int(labels[ind])
                if label in [4,9]:
                    continue
                score = scores[ind]
                det3d = Detection3D()
                det3d.header.frame_id = 'odom'

                pose = Pose()
                pose.position.x = bbox[0].item()
                pose.position.y = bbox[1].item()
                pose.position.z = bbox[2].item()

                quat = Quaternion()
                q = tf_transformations.quaternion_from_euler(0, 0, bbox[-1].item())
                quat.x = q[0]
                quat.y = q[1]
                quat.z = q[2]
                quat.w = q[3]
                pose.orientation = quat

                dimensions = Vector3()
                dimensions.x = bbox[3].item()
                dimensions.y = bbox[4].item()
                dimensions.z = bbox[5].item()

                det3d.bbox.center = pose
                det3d.bbox.size = dimensions
                object_hypothesis = ObjectHypothesisWithPose()
                object_hypothesis.hypothesis.class_id = self.class_names[label]
                object_hypothesis.hypothesis.score = score.item()
                det3d.results.append(object_hypothesis)
                
                det3d_array.detections.append(det3d)
                
                self.logger.debug(f"Found class_id: {self.class_names[label]}, score: {score.item()}")
        
            self.marker_pub.publish(det3d_array)
        
def main(args=None):
    rclpy.init(args=args)
    infer_node = InferNode()
    rclpy.spin(infer_node)
    infer_node.destroy_node()
    rclpy.shutdown()
