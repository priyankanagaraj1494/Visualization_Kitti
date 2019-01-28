import numpy as np
import open3d
import sys
import os
import math
import cv2
from helper_functions import read_calib_file, read_label, read_gt,rotate_image
from vis_utils import create3Dbbox,create3dBbox_images, create3dBbox_on_lidar_images,create_gt_2d_bbox
from video_generator import video_generator
import parameter

#-------- Get index -------------------

#predict_dir = "/home/kangning/PycharmProjects/visualization/predict/"
#image_dir = "/home/kangning/PycharmProjects/visualization/image/"
#lidar_dir = "/home/kangning/PycharmProjects/visualization/velodyne/"
#calib_dir = "/home/kangning/PycharmProjects/visualization/calib/"
#trajectory_dir = "/home/kangning/PycharmProjects/visualization/camera_trajectory/"
#lidar_image_dir = "/home/kangning/PycharmProjects/visualization/new_image/"
#label_dir = "/home/kangning/PycharmProjects/visualization/gt/"


predict_dir = "C:/Users/Pnagaraj/Lidar/dataset/training/velodyne1/"
#predict_dir = "/home/kangning/Desktop/detection_results_v1_299e_wopcnn/data/"
#predict_dir = "/home/kangning/Documents/Masterarbeit/frus_exp_wo_pointcnn/train/log_v1_out_3/eval_kitti/build/results/val_add_edgeConv/data/"
image_dir = "C:/Users/Pnagaraj/Lidar/dataset/training/image/"
lidar_dir = "C:/Users/Pnagaraj/Lidar/dataset/training/velodyne1/"
calib_dir = "C:/Users/Pnagaraj/Lidar/dataset/training/calib/"
lidar_image_dir = "C:/Users/Pnagaraj/Lidar/dataset/training/top/"
trajectory_dir = "C:/Users/Pnagaraj/Lidar/visualization/camera_trajectory/"
label_dir = "C:/Users/Pnagaraj/Lidar/dataset/training/label/"

def ScaleRows(z):
    return int((height/2) + math.floor(z* (height / (2* roi_height))))   # Scaling z from 0 to 850

def ScaleCols(x):
    return int((width/2) + math.floor(x*(width / (2* roi_width))))   # Scaling x from 0 to 1700
def ScaleY(y):
    return math.floor((y +3) *25)

predict_names = os.listdir(predict_dir)



def draw_geometries_dark_background(geometries):
    vis=open3d.Visualizer()
    vis.create_window()
    opt=vis.get_render_option()
    opt.background_color=np.asarray([0,0,0])
    for geometry in geometries:
        vis.add_geometry(geometry)
    vis.run()
    vis.destroy_window()

def draw_3d_bbox_on_image(img, bbox_list_on_images):
    image = np.copy(img)
    #image= cv2.cvtColor (image,cv2.COLOR_BGR2RGB)
    for bbox in bbox_list_on_images:
        for line in bbox['lines']:
            if bbox['colors']==[255,0,0]:
                bbox['colors']=[0,0,255]
            color = bbox['colors']

            point_in_3d = bbox['points'][line].T


            ###Project to Image
            #p3d = np.vstack((bbox['points'][line].T, np.ones((1, bbox['points'][line].shape[0]))))
            ''' Project 3d points to image plane.
               Usage: pts_2d = projectToImage(pts_3d, P)
                 input: pts_3d: nx3 matrix
                        P:      3x4 projection matrix
                 output: pts_2d: nx2 matrix
                 P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
                 => normalize projected_pts_2d(2xn)
                 <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
                     => normalize projected_pts_2d(nx2)
               '''
            point_in_3d = np.vstack((bbox['points'][line].T, np.ones((1, bbox['points'][line].shape[0]))))


            point_in_2d = np.dot(bbox['calib_matrix'],point_in_3d)

            for m, point in enumerate(point_in_2d[2 , :]):
                point_in_2d[:, m] = point_in_2d[ :, m] / point


            point_in_2d = np.int32([point_in_2d[:2, :].T])

            point_0 = point_in_2d[0][0]
            point_1 = point_in_2d[0][1]
            cv2.line(image, (point_0[0],point_0[1]),(point_1[0],point_1[1]),color, 2)

    return image

def draw_gt_2d_bbox_on_image(img, gt_bbox_list):
    image= np.copy(img)
    for gt_bbox in gt_bbox_list:
        for line in gt_bbox["lines"]:
            if gt_bbox['colors']==[255,0,0]:
                gt_bbox['colors']=[0,0,255]
            color = gt_bbox['colors']
            points = gt_bbox["gt_2d"][line]
            point_0 = points[0]
            point_1 = points[1]

            cv2.line(image, (point_0[0],point_0[1]),(point_1[0],point_1[1]),color,2)

    return image

def draw_3d_bbox_on_lidar_image(lidar_image, bbox_list_on_lidar_image):
    image = np.copy(lidar_image)
    for bbox in bbox_list_on_lidar_image:
        for line in bbox['lines']:
            if bbox['colors'] ==[255,0,0]:
                bbox['colors']=[0,0,255]
            color = bbox['colors']
            point_in_2d = bbox['points'][line]


            point_0 = point_in_2d[0]
            point_1 = point_in_2d[1]


            cv2.line(image, (point_0[0], point_0[1]),(point_1[0], point_1[1]), color, 2)

    return image

image_ids = []
#for index,predict_name in enumerate(sorted(predict_names)):
for predict_name in ['000849.txt']:

    sequence = predict_name.split(".txt")[0]
    image_ids.append(sequence)
    img_path = image_dir + sequence + ".png"
    lidar_image_path =  lidar_image_dir + sequence + ".png"
    image = cv2.imread(img_path,-1)
    lidar_image = cv2.imread(lidar_image_path , -1)
    lidar_path = lidar_dir + sequence + ".bin"
    #print(lidar_path)
    point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)  # point cloud channel is x,y,z,reflectance
    #print(type(point_cloud))
    #### Remove points that locate behind camera
    point_cloud = point_cloud[point_cloud[:, 0] > -2.5, :]
    calib_path = calib_dir + sequence + ".txt"
    camera_trajectory =  trajectory_dir + sequence + ".json"

    calib = read_calib_file(calib_path)
    P2 = calib['P2']

    Tr_velo_to_cam_original = calib['Tr_velo_to_cam']
    R0_rect_original = calib['R0_rect']

    R0_rect = np.eye(4)

    '''
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])

    '''

    R0_rect[0:3, 0:3] = R0_rect_original

    Tr_velo_to_cam = np.eye(4)

    Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_original

    #### Tranform the points into rectified camera coordiantes ##########################################################################

    # tr_velo_to_cam dot r0_rect
    # using homogeneous transformation

    point_cloud_xyz = point_cloud[:, 0:3]
    print(point_cloud_xyz.shape)
    point_cloud_xyz_homo = np.ones((point_cloud.shape[0], 4))
    point_cloud_xyz_homo[:, 0:3] = point_cloud[:, 0:3]
    point_cloud_camera_non_rec = np.dot(Tr_velo_to_cam, point_cloud_xyz_homo.T)
    point_cloud_camera_rect = np.dot(R0_rect, point_cloud_camera_non_rec).T  # 4 channels, homogeneous coordinates

    # Homogeneous to cartesian: convert point_cloud_camera_rect(homogeneous coordinates into cartesian coordinates/point_cloud_xyz_camera )

    point_cloud_xyz_camera = np.zeros((point_cloud_camera_rect.shape[0], 3))  # 3 channels , cartesian coordinates
    point_cloud_xyz_camera[:, 0] = point_cloud_camera_rect[:, 0] / point_cloud_camera_rect[:, 3]
    point_cloud_xyz_camera[:, 1] = point_cloud_camera_rect[:, 1] / point_cloud_camera_rect[:, 3]
    point_cloud_xyz_camera[:, 2] = point_cloud_camera_rect[:, 2] / point_cloud_camera_rect[:, 3]

    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(point_cloud_xyz_camera)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    #open3d.draw_geometries([point_cloud_xyz_camera])
    label_path = predict_dir + sequence + ".txt"

    ###read groundtruth
    gt_path = label_dir + sequence + ".txt"
    gt_dics = read_gt(gt_path,sequence)
    gt_bboxes = []

    for Bbox_2d in gt_dics[sequence]:

        xmin = Bbox_2d["xmin"]
        ymin = Bbox_2d["ymin"]
        xmax = Bbox_2d["xmax"]
        ymax = Bbox_2d["ymax"]

        gt_bbox= create_gt_2d_bbox(xmin=xmin,ymin=ymin,xmax=xmax,ymax=ymax,type_list=Bbox_2d["class_id"],color_dicts=parameter.color_dictionary)
        gt_bboxes.append(gt_bbox)


    eval_dicts = read_label(label_path,sequence)



    Predic_bboxes = []
    Predict_bboxes_images = []
    Predict_bboxes_on_lidar_images = []

    for Bbox in eval_dicts[sequence]:
        centers = Bbox["center"]
        width = Bbox["width"]
        height = Bbox["height"]
        length = Bbox["length"]
        r_y = Bbox["r_y"]
        class_id = Bbox["class_id"]

        predict_bbox= create3Dbbox(center=centers, h=height, w= width, l= length, r_y= r_y, type_list=class_id,color_dicts = parameter.color_dictionary)
        Predic_bboxes+= predict_bbox

        predict_bbox_image = create3dBbox_images(center=centers,h=height,w=width,l=length,r_y=r_y,calib_matrix=P2,type_list=class_id, color_dicts = parameter.color_dictionary)
        Predict_bboxes_images.append(predict_bbox_image)


        predict_bbox_on_lidar_image = create3dBbox_on_lidar_images(center=centers, h=height, w= width, l= length, r_y= r_y,type_list=class_id, color_dicts = parameter.color_dictionary)
        Predict_bboxes_on_lidar_images.append(predict_bbox_on_lidar_image)


    #image_with_3d_bbox = draw_3d_bbox_on_image(image,Predict_bboxes_images)


   # draw_geometries_dark_background(Predic_bboxes + [pcd])

    #image_for_pcds = video_generator()
    #image_for_point_cloud_wo_bbox = image_for_pcds.create_image([pcd])
    #image_for_point_cloud = image_for_pcds.create_image(Predic_bboxes + [pcd])

    image_for_lidar_with_3d_bbox = draw_3d_bbox_on_lidar_image(lidar_image, Predict_bboxes_on_lidar_images)
    new_image= np.zeros((850,850,3),np.uint8)
    new_image = image_for_lidar_with_3d_bbox[(0):(849), (849):(1699), :]
    top = cv2.resize(new_image, dsize=(parameter.top_view_lidar_width, parameter.top_view_lidar_height))
    rotate_top_view_lidar_image= rotate_image(top,angle= 90)



    image_with_gt_2d_bbox = draw_gt_2d_bbox_on_image(image, gt_bboxes)
    #cv2.imwrite("/home/kangning/Desktop/image_with_gt_2d_bbox/" + sequence + ".png", image_with_gt_2d_bbox)

    # generate the final pretty output frame

    #Output_frame = image_for_point_cloud

    #height = parameter.image_height_2
    #width = parameter.image_width_2

    #thumnail_3d_bbox = cv2.resize(image_with_3d_bbox, dsize = (width, height))
    #thumbnail_2d_gt_bbox = cv2.resize(image_with_gt_2d_bbox,dsize = (width, height))


    #Output_frame[(1026-parameter.offset_y-height): (1026 -parameter.offset_y),  parameter.margin + parameter.offset_x: parameter.margin+parameter.offset_x + width, :] = thumnail_3d_bbox
    #Output_frame[(1026-parameter.offset_y-height): (1026 -parameter.offset_y),  parameter.margin+ 2*parameter.offset_x + width :parameter.margin+ 2*parameter.offset_x + width +parameter.top_view_lidar_width, : ]= rotate_top_view_lidar_image
    #Output_frame[(1026-parameter.offset_y-height): (1026 -parameter.offset_y), parameter.margin+ 3*parameter.offset_x +  width +parameter.top_view_lidar_width: parameter.margin+ 3*parameter.offset_x +  2 *width +parameter.top_view_lidar_width, :] = thumbnail_2d_gt_bbox

    #Output_frame[(1026-parameter.offset_y-parameter.image_height_2): (1026 -parameter.offset_y), parameter.offset_x:parameter.offset_x+parameter.image_width_2, :]= thumnail_3d_bbox
    #Output_frame[(1026-parameter.offset_y-parameter.image_height_2): (1026 -parameter.offset_y), 2*parameter.offset_x+parameter.image_width_2: 2*parameter.offset_x+parameter.image_width_2*2,:]= thumbnail_2d_gt_bbox


    #Output_frame[0:parameter.top_view_lidar_height,0:parameter.top_view_lidar_width,:]=rotate_top_view_lidar_image

    #if Output_frame is None:
    #    print ("Error in writing frame")

    path_dir="/home/kangning/Desktop/2d/"
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    #path = os.path.join("/home/kangning/Desktop/3D/", str(sequence) + ".png")


    cv2.imwrite(path_dir+ sequence +".png", image_with_gt_2d_bbox)
























