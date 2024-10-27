# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#

import sys
# from xmlrpc.client import FastParser

sys.path.append('/is/ps2/otaheri/code_repos/GRIP_adobe/project')
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import os, glob
import smplx
# from smplx import SMPLXLayer

import argparse
import shutil
import open3d as o3d
# import time
from datetime import datetime
from tqdm import tqdm

from tools.objectmodel import ObjectModel
from tools.cfg_parser import Config

from tools.utils import makepath, makelogger
from tools.utils import parse_npz
from tools.utils import params2torch
from tools.utils import prepare_params
# from tools.utils import to_cpu, to_np, to_tensor
from tools.utils import to_cpu, to_tensor
from tools.utils import append2dict
# from tools.utils import np2torch, torch2np
from tools.utils import torch2np
# from tools.utils import aa2rotmat, rotmat2aa, rotate, rotmul, euler
from tools.utils import aa2rotmat, rotmat2aa, rotate, rotmul
from tools.utils import smplx_loc2glob

import trimesh
import pickle

# from tools.vis_tools import points_to_spheres, sp_animation_old, get_ground
# from psbody.mesh.colors import name_to_rgb

from bps_torch.bps import bps_torch
from bps_torch.tools import sample_uniform_cylinder, sample_sphere_uniform, sample_hemisphere_uniform
# from psbody.mesh import Mesh, MeshViewers, MeshViewer
# from psbody.mesh import Mesh

from my_utils.transforms.rotation_transform import (
    axis_angle_to_rotation_6d,
    local2global_pose
)  
if os.name == 'nt':
    from pytorch3d.pytorch3d.transforms.rotation_conversions import (
    matrix_to_rotation_6d,
    axis_angle_to_matrix,
    matrix_to_quaternion,
    quaternion_apply
    )
else:
    from pytorch3d.transforms.rotation_conversions import (
    matrix_to_rotation_6d,
    axis_angle_to_matrix,
    matrix_to_quaternion,
    quaternion_apply
    )

# from models.model_utils import full2bone, full2bone_aa, parms_6D2full, parms_decode_full

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INTENTS = ['lift', 'pass', 'offhand', 'use', 'all']


class MNetDataSet(object):

    def __init__(self, cfg, logger=None, **params):

        self.cfg = cfg
        self.grab_path = cfg.grab_path
        self.out_path = cfg.out_path
        self.cwd = os.path.dirname(sys.argv[0])
        makepath(self.out_path)

        if logger is None:
            log_dir = os.path.join(self.out_path, 'grab_preprocessing.log')
            self.logger = makelogger(log_dir=log_dir, mode='a').info
        else:
            self.logger = logger
        self.logger('Starting data preprocessing !')

        # assert cfg.intent in INTENTS

        self.intent = cfg.intent
        self.logger('intent:%s --> processing %s sequences!' % (self.intent, self.intent))

        if cfg.splits is None:
            self.splits = {'test': .1,
                           'val': .05,
                           'train': .85}
        else:
            assert isinstance(cfg.splits, dict)
            self.splits = cfg.splits

        self.all_seqs = glob.glob(os.path.join(self.grab_path, 'grab/*/*.npz'))

        ### to be filled 
        self.selected_seqs = []
        self.obj_based_seqs = {}
        self.sbj_based_seqs = {}
        self.split_seqs = {'test': [],
                           'val': [],
                           'train': []
                           }
        
        ##############     From XRZ    ##############
        example_smplx_path = os.path.join(cfg.model_path, 'smplx', 'SMPLX_NEUTRAL.pkl')
        with open(example_smplx_path, 'rb') as f:
            self.example_smplx = pickle.load(f, encoding='latin1')
        self.kintree = self.example_smplx['kintree_table'][0]

        ### group, mask, and sort sequences based on objects, subjects, and intents
        self.process_sequences()

        self.logger('Total sequences: %d' % len(self.all_seqs))
        self.logger('Selected sequences: %d' % len(self.selected_seqs))
        self.logger('Number of sequences in each data split : train: %d , test: %d , val: %d'
                    % (len(self.split_seqs['train']), len(self.split_seqs['test']), len(self.split_seqs['val'])))

        ### process the data
        self.data_preprocessing(cfg)

    def data_preprocessing(self, cfg):

        self.obj_info = {}
        self.sbj_info = {}

        bps_path = makepath(os.path.join(cfg.out_path, 'bps.pt'), isfile=True)
        bps_orig_path = f'{self.cwd}/../data/bps.pt'

        self.bps_torch = bps_torch()

        R_bps = torch.tensor(
            [[1., 0., 0.],
             [0., 0., -1.],
             [0., 1., 0.]]).reshape(1, 3, 3).to(device)
        if os.path.exists(bps_path):
            self.bps = torch.load(bps_path)
            self.logger(f'loading bps from {bps_path}')
        elif os.path.exists(bps_orig_path):
            self.bps = torch.load(bps_orig_path)
            self.logger(f'loading bps from {bps_orig_path}')
        else:
            self.bps_obj = sample_sphere_uniform(n_points=cfg.n_obj, radius=cfg.r_obj).reshape(1, -1, 3)
            self.bps_sbj = rotate(
                sample_uniform_cylinder(n_points=cfg.n_sbj, radius=cfg.r_sbj, height=cfg.h_sbj).reshape(1, -1, 3),
                R_bps.transpose(1, 2))
            self.bps_rh = sample_hemisphere_uniform(n_points=cfg.n_rh, radius=cfg.r_rh, axis='-y').reshape(1, -1, 3)
            self.bps_lh = sample_hemisphere_uniform(n_points=cfg.n_rh, radius=cfg.r_rh, axis='-y').reshape(1, -1, 3)
            self.bps_hd = sample_sphere_uniform(n_points=cfg.n_hd, radius=cfg.r_hd).reshape(1, -1, 3)

            self.bps = {
                'obj': self.bps_obj.cpu(),
                'sbj': self.bps_sbj.cpu(),
                'rh': self.bps_rh.cpu(),
                'lh': self.bps_lh.cpu(),
                'hd': self.bps_hd.cpu(),
            }
            torch.save(self.bps, bps_path)

        # vertex_label_contact = to_tensor(np.load(f'{self.cwd}/../consts/vertex_label_contact.npy'), dtype=torch.int8).reshape(1, -1)
        verts_ids = to_tensor(np.load(f'{self.cwd}/../consts/verts_ids_0512.npy'), dtype=torch.long)
        rh_verts_ids = to_tensor(np.load(f'{self.cwd}/../consts/rhand_smplx_ids.npy'), dtype=torch.long)
        lh_verts_ids = to_tensor(np.load(f'{self.cwd}/../consts/lhand_smplx_ids.npy'), dtype=torch.long)

        rh_faces_ids = to_tensor(np.load(f'{self.cwd}/../consts/rhand_faces.npy'), dtype=torch.long)
        lh_faces_ids = to_tensor(np.load(f'{self.cwd}/../consts/lhand_faces.npy'), dtype=torch.long)

        rh_ids_sampled = torch.tensor(np.where([id in rh_verts_ids for id in verts_ids])[0]).to(torch.long)
        lh_ids_sampled = torch.tensor(np.where([id in lh_verts_ids for id in verts_ids])[0]).to(torch.long)


        stime = datetime.now().replace(microsecond=0)
        shutil.copy2(sys.argv[0],
                     os.path.join(self.out_path,
                                  os.path.basename(sys.argv[0]).replace('.py', '_%s.py' % datetime.strftime(stime,
                                                                                                            '%Y%m%d_%H%M'))))

        self.subject_mesh = {}
        self.obj_info = {}
        self.sbj_info = {}

        for split in self.split_seqs.keys():
        # for split in ['train', 'test', 'val']:
            # split = 'train'
            outfname = makepath(os.path.join(cfg.out_path, split, 'grasp_motion_data.npy'), isfile=True)

            if os.path.exists(outfname):
                self.logger('Results for %s split already exist.' % (split))
                self.logger('Loading subject and object infos!')
                self.obj_info = np.load(os.path.join(self.out_path, 'obj_info.npy'))
                self.sbj_info = np.load(os.path.join(self.out_path, 'sbj_info.npy'))
                continue
            else:
                self.logger('Processing data for %s split.' % (split))

            frame_names = []

            grasp_motion_data = {
                'transl': [],
                'transl_p': [],
                'fullpose': [],
                'fullpose_p': [],
                'fullpose_rotmat': [],
                'fullpose_rotmat_p': [],
                'fullpose_p_arms': [],
                'fullpose_rotmat_p_arms': [],

                'verts': [],
                'verts_h': [],
                'verts_p_arms': [],
                'velocity': [],

                'transl_obj': [],
                'global_orient_obj': [],
                'global_orient_rotmat_obj': [],

                'bps_obj_glob': [],
                'bps_obj_rh': [],
                'bps_obj_lh': [],
                'bps_obj_rh_p_arms': [],
                'bps_obj_lh_p_arms': [],

                'rh2obj_gt': [],
                'rh2obj_h': [],
                'rh2obj_p': [],
                'rh2obj_p_arms': [],
                'rh2obj_h_vel': [],
                'rh2obj_p_arms_vel': [],

                'lh2obj_gt': [],
                'lh2obj_h': [],
                'lh2obj_p': [],
                'lh2obj_p_arms': [],
                'lh2obj_h_vel': [],
                'lh2obj_p_arms_vel': [],

                'lh_contact': [],
                'rh_contact': [],


                'rot_rh2obj': [],
                'rot_lh2obj': [],
                'trans_rh2obj': [],
                'trans_lh2obj': [],

                'full_seq_id': [],
                'rel_rot': [],
                'rel_trans': [],
                'frame_ids':[],
                'frame_ends': [],

                'lh2obj_h_ids':[],
                'rh2obj_h_ids':[],

                'lh2obj_gt_ids':[],
                'rh2obj_gt_ids':[],

            }

            ##############     From XRZ    ####################
            normal_motion_data = {
                'rotation_local_body_gt_list': [],
                'rotation_local_left_hand_gt_list': [],
                'rotation_local_right_hand_gt_list': [],
                'hmd_position_global_gt_list': [],
                'smplx_part_params': [],
                'body_pos_global_gt_world': [],
                'left_hand_pos_global_gt_world': [],
                'right_hand_pos_global_gt_world': [],
                'head_global_trans_list': []
            }

            hoi_obj_data = {
                'lh_labels': [],
                'lh_obj_dists': [],
                'lh_obj_rep': [],
                'rh_labels': [],
                'rh_obj_dists': [],
                'rh_obj_rep': []
            }

            ##############     From XRZ    ####################

            for seq_i, sequence in enumerate(tqdm(self.split_seqs[split])):

                seq_data = parse_npz(sequence)

                obj_name = seq_data.obj_name
                sbj_id = seq_data.sbj_id

                n_comps = seq_data.n_comps
                gender = seq_data.gender

                # frame_mask, grasp_frame = self.filter_goal_frames(seq_data)
                frame_mask = np.array([True if i % 4 == 0 else False for i in range(seq_data.n_frames)])

                T = frame_mask.sum()
                if T < 2:
                    print('Warning: sequence has less than 2 frames, skip it')
                    continue  # if no frame is selected continue to the next sequence

                ##### motion data preparation
                sbj_vtemp = self.load_sbj_verts(sbj_id, seq_data)
                obj_info = self.load_obj_verts(obj_name, seq_data, cfg.n_verts_sample)

                sbj_params = prepare_params(seq_data.body.params, frame_mask)
                obj_params = prepare_params(seq_data.object.params, frame_mask)
                # contact_data_orig = seq_data.contact.body[frame_mask]

                sbj_params_orig = params2torch(sbj_params)
                obj_params_orig = params2torch(obj_params)

                ################# for chunks
                bs = T

                with torch.no_grad():
                    sbj_m = smplx.create(model_path=cfg.model_path,
                                         model_type='smplx',
                                         gender=gender,
                                         num_pca_comps=n_comps,
                                         v_template=sbj_vtemp,
                                         batch_size=bs-1)

                    sbj_m_h = smplx.create(model_path=cfg.model_path,
                                           model_type='smplx',
                                           gender=gender,
                                           num_pca_comps=n_comps,
                                           v_template=sbj_vtemp,
                                           flat_hand_mean=False,
                                           batch_size=bs-1)
                    
                    sbj_m_fhand = smplx.create(
                        model_path=cfg.model_path,
                        model_type='smplx',
                        gender=gender,
                        use_pca=False,
                        v_template=sbj_vtemp,
                        batch_size=bs
                    )
                
                    obj_m = ObjectModel(v_template=obj_info['verts_sample'],
                                        batch_size=bs-1)

                    root_offset = smplx.lbs.vertices2joints(sbj_m.J_regressor, sbj_m.v_template.view(1, -1, 3))[0, 0]

                    # motion_sbj = sbj_params_orig
                    motion_sbj = {k: v.clone()[1:] for k, v in sbj_params_orig.items()}
                    motion_sbj['fullpose_rotmat'] = aa2rotmat(motion_sbj['fullpose'])

                    ##############     From XRZ    ####################
                    fullpose = sbj_params_orig['fullpose']

                    global_orient = fullpose[:, 0:3]
                    body_pose = fullpose[:, 3:66]
                    left_hand_pose = fullpose[:, 75:120]
                    right_hand_pose = fullpose[:, 120:165]

                    fhand_params = {
                        'transl': sbj_params_orig['transl'],
                        'global_orient': global_orient,
                        'body_pose': body_pose,
                        'left_hand_pose': left_hand_pose,
                        'right_hand_pose': right_hand_pose
                    }
                    sbj_output_fhand = sbj_m_fhand(**fhand_params)
                    # verts_sbj_fhand = sbj_output_fhand.vertices
                    joints_sbj_fhand = sbj_output_fhand.joints
                    torso_pose = torch.cat([global_orient, body_pose], dim=-1)
                    torso_pose_aa = torso_pose[..., :66].reshape(-1, 22, 3)
                    torso_pose_6d = axis_angle_to_rotation_6d(torso_pose_aa).reshape(bs, -1)
                    rotation_local_body_gt_list = torso_pose_6d[1:]

                    left_hand_pose_aa = left_hand_pose[...,:45].reshape(-1, 15, 3)
                    left_hand_pose_6d = axis_angle_to_rotation_6d(left_hand_pose_aa).reshape(bs, -1)
                    rotation_local_left_hand_gt_list = left_hand_pose_6d[1:]

                    right_hand_pose_aa = right_hand_pose[...,:45].reshape(-1, 15, 3)
                    right_hand_pose_6d = axis_angle_to_rotation_6d(right_hand_pose_aa).reshape(bs, -1)
                    rotation_local_right_hand_gt_list = right_hand_pose_6d[1:]

                    rotation_local_body_matrot = axis_angle_to_matrix(torso_pose_aa).reshape(bs, -1, 9)
                    rotation_global_body_matrot = local2global_pose(rotation_local_body_matrot, self.kintree[:22])
                    head_rotation_global_matrot = rotation_global_body_matrot[:, [15], :, :]

                    rotation_global_body_6d = matrix_to_rotation_6d(
                        rotation_global_body_matrot
                    ).reshape(rotation_global_body_matrot.shape[0], -1, 6)
                    input_rotation_global_body_6d = rotation_global_body_6d[1:, [15, 20, 21], :]

                    rotation_body_velocity_global_matrot = torch.matmul(
                        torch.inverse(rotation_global_body_matrot[:-1]),
                        rotation_global_body_matrot[1:]
                    )

                    rotation_body_velocity_global_6d = matrix_to_rotation_6d(
                        rotation_body_velocity_global_matrot
                        ).reshape(rotation_body_velocity_global_matrot.shape[0], -1, 6)
                    input_body_rotation_velocity_global_6d = rotation_body_velocity_global_6d[
                        :, [15, 20, 21], :
                        ]
                    
                    body_pos_global_gt_world = joints_sbj_fhand[:, :22, :]
                    left_hand_pos_global_gt_world = joints_sbj_fhand[:, 25:40, :]
                    right_hand_pos_global_gt_world = joints_sbj_fhand[:, 40:55, :]
                    head_pos_world = joints_sbj_fhand[:, 15, :]

                    head_global_trans = torch.eye(4).repeat(
                        head_pos_world.shape[0], 1, 1
                    )
                    head_global_trans[:, :3, :3] = head_rotation_global_matrot.squeeze()
                    head_global_trans[:, :3, 3] = body_pos_global_gt_world[:, 15, :]
                    head_global_trans_list = head_global_trans[1:]

                    num_frames = body_pos_global_gt_world.shape[0] - 1

                    hmd_position_global_gt_list = torch.cat(
                        [
                            input_rotation_global_body_6d.reshape(num_frames, -1),
                            input_body_rotation_velocity_global_6d.reshape(num_frames, -1),
                            body_pos_global_gt_world[1: , [15, 20, 21], :].reshape(num_frames, -1),
                            body_pos_global_gt_world[1: , [15, 20, 21], :].reshape(
                                num_frames, -1
                                ) - 
                            body_pos_global_gt_world[:-1, [15, 20, 21], :].reshape(
                                num_frames, -1
                                ),
                        ],
                        dim=-1
                    )
                    normal_motion_data['rotation_local_body_gt_list'].append(rotation_local_body_gt_list.cpu().float())
                    normal_motion_data['rotation_local_left_hand_gt_list'].append(rotation_local_left_hand_gt_list.cpu().float())
                    normal_motion_data['rotation_local_right_hand_gt_list'].append(rotation_local_right_hand_gt_list.cpu().float())
                    normal_motion_data['hmd_position_global_gt_list'].append(hmd_position_global_gt_list.cpu().float())
                    normal_motion_data['body_pos_global_gt_world'].append(body_pos_global_gt_world[1:].cpu().float())
                    normal_motion_data['left_hand_pos_global_gt_world'].append(left_hand_pos_global_gt_world[1:].cpu().float())
                    normal_motion_data['right_hand_pos_global_gt_world'].append(right_hand_pos_global_gt_world[1:].cpu().float())
                    normal_motion_data['head_global_trans_list'].append(head_global_trans_list.cpu().float())

                    normal_motion_data['smplx_part_params'].append(fhand_params)


                    motion_obj = {k: v.clone()[1:] for k, v in obj_params_orig.items()}

                    ##############     From XRZ    ####################

                    # motion_obj = obj_params_orig


                    sbj_output = sbj_m(**motion_sbj)
                    verts_sbj = sbj_output.vertices
                    joints_sbj = sbj_output.joints

                    motion_sbj_h = {k: v.clone() for k, v in motion_sbj.items() if not 'hand' in k}

                    sbj_output_h = sbj_m_h(**motion_sbj_h)
                    verts_sbj_h = sbj_output_h.vertices
                    joints_sbj_h = sbj_output_h.joints

                    obj_out = obj_m(**motion_obj)
                    verts_obj = obj_out.vertices

                    motion_sbj_p = {k: v.clone() for k, v in motion_sbj.items()}
                    motion_sbj_p_arms = {k: v.clone() for k, v in motion_sbj.items() if not 'hand' in k}
                    #### add wrist perturbation
                    aug_rw_p = np.random.randn(T-1, 3) * 0.04
                    aug_lw_p = np.random.randn(T-1, 3) * 0.04
                    motion_sbj_p['body_pose'][:,3*18:3*19] += to_tensor(aug_rw_p)
                    motion_sbj_p['body_pose'][:,3*19:3*20] += to_tensor(aug_lw_p)

                    aug_rw_p = np.random.randn(T-1, 3) * 0.03
                    aug_lw_p = np.random.randn(T-1, 3) * 0.03

                    motion_sbj_p_arms['body_pose'][:,3*18:3*19] += to_tensor(aug_rw_p)
                    motion_sbj_p_arms['body_pose'][:,3*19:3*20] += to_tensor(aug_lw_p)

                    #### add arm perturbation
                    aug_ra_p = np.random.randn(T-1, 3*2) * 0.01
                    # aug_la_p = np.random.randn(T, 3) * 0.01
                    motion_sbj_p_arms['body_pose'][:,3*16:3*18] += to_tensor(aug_ra_p)
                    # motion_sbj_p['body_pose'][:,3*19:3*20] += to_tensor(aug_lw_p)

                    ####
                    aug_rh_p = np.random.randn(T-1, 24) * 0.3
                    aug_lh_p = np.random.randn(T-1, 24) * 0.3
                    
                    motion_sbj_p['right_hand_pose'] += to_tensor(aug_rh_p)
                    motion_sbj_p['left_hand_pose'] += to_tensor(aug_lh_p)

                    # motion_sbj_p['fullpose'][:,]

                
                    # motion_sbj_h['right_hand_pose'] = rh_pose.reshape(1,15,3,3).repeat(T,1,1,1)
                    sbj_output_p = sbj_m_h(**motion_sbj_p_arms, return_full_pose=True)
                    verts_sbj_p = sbj_output_p.vertices
                    joints_sbj_p = sbj_output_p.joints

                    fullpose_p = sbj_output_p.full_pose
                    motion_sbj_p['fullpose'] = fullpose_p.clone()
                    motion_sbj_p['fullpose_rotmat'] = aa2rotmat(fullpose_p)

                    sbj_output_p_arms = sbj_m_h(**motion_sbj_p_arms, return_full_pose=True)
                    verts_sbj_p_arms = sbj_output_p_arms.vertices
                    joints_sbj_p_arms = sbj_output_p_arms.joints

                    fullpose_p_arms = sbj_output_p_arms.full_pose
                    motion_sbj_p_arms['fullpose'] = fullpose_p_arms.clone()
                    motion_sbj_p_arms['fullpose_rotmat'] = aa2rotmat(fullpose_p_arms)

                    sbj_in = {k: to_cpu(v.clone()) for k, v in motion_sbj.items()}
                    sbj_in_p = {k + '_p': to_cpu(v.clone()) for k, v in motion_sbj_p.items()}
                    sbj_in_p_arms = {k + '_p_arms': to_cpu(v.clone()) for k, v in motion_sbj_p_arms.items()}
                    obj_in = {k + '_obj': to_cpu(v.clone()) for k, v in motion_obj.items()}
                    obj_in['global_orient_rotmat_obj'] = aa2rotmat(motion_obj['global_orient'])

                    append2dict(grasp_motion_data, sbj_in)
                    append2dict(grasp_motion_data, sbj_in_p)
                    append2dict(grasp_motion_data, sbj_in_p_arms)
                    append2dict(grasp_motion_data, obj_in)

                    verts_sampled = verts_sbj[:, verts_ids].clone()
                    grasp_motion_data['verts'].append(to_cpu(verts_sampled))

                    verts_sampled = verts_sbj_h[:, verts_ids].clone()
                    grasp_motion_data['verts_h'].append(to_cpu(verts_sampled))

                    verts_sampled = verts_sbj_p_arms[:, verts_ids].clone()
                    grasp_motion_data['verts_p_arms'].append(to_cpu(verts_sampled))


                    bps_type_all = ['dists','deltas','closest']
                    bps_type = 'dists'

                    R_rh_glob = smplx_loc2glob(motion_sbj_h['fullpose_rotmat'])[:, 21]
                    rh_bps = rotate(self.bps['rh'], R_rh_glob)

                    rh_bps = rh_bps + joints_sbj_h[:, 43:44] #flat hand


                    bps_obj_rh = self.bps_torch.encode(x=verts_obj,
                                                       feature_type=[bps_type],
                                                       custom_basis=rh_bps)

                    grasp_motion_data['bps_obj_rh'].append(to_cpu(bps_obj_rh[bps_type]))


                    R_lh_glob = smplx_loc2glob(motion_sbj_h['fullpose_rotmat'])[:, 20]
                    lh_bps = rotate(self.bps['lh'], R_lh_glob)

                    lh_bps = lh_bps + joints_sbj_h[:, 28:29] #flat hand

                    bps_obj_lh = self.bps_torch.encode(x=verts_obj,
                                                       feature_type=[bps_type],
                                                       custom_basis=lh_bps)[bps_type]

                    grasp_motion_data['bps_obj_lh'].append(to_cpu(bps_obj_lh))

                    ################# ARMS
                    R_rh_glob_p_arms = smplx_loc2glob(motion_sbj_p_arms['fullpose_rotmat'])[:, 21]
                    rh_bps_p_arms = rotate(self.bps['rh'], R_rh_glob_p_arms)

                    rh_bps_p_arms = rh_bps_p_arms + joints_sbj_p_arms[:, 43:44] #flat hand

                    bps_obj_rh_p_arms = self.bps_torch.encode(x=verts_obj,
                                                         feature_type=[bps_type],
                                                         custom_basis=rh_bps_p_arms)
                    
                    grasp_motion_data['bps_obj_rh_p_arms'].append(to_cpu(bps_obj_rh_p_arms[bps_type]))

                    R_lh_glob_p_arms = smplx_loc2glob(motion_sbj_p_arms['fullpose_rotmat'])[:, 20]
                    lh_bps_p_arms = rotate(self.bps['lh'], R_lh_glob_p_arms)

                    lh_bps_p_arms = lh_bps_p_arms + joints_sbj_p_arms[:, 28:29] #flat hand

                    bps_obj_lh_p_arms = self.bps_torch.encode(x=verts_obj,
                                                            feature_type=[bps_type],
                                                            custom_basis=lh_bps_p_arms)[bps_type]
                    
                    grasp_motion_data['bps_obj_lh_p_arms'].append(to_cpu(bps_obj_lh_p_arms))


                    rh2obj_gt = self.bps_torch.encode(x= verts_obj,
                                                     feature_type=bps_type_all,
                                                     custom_basis=verts_sbj[:, verts_ids[rh_ids_sampled]])
                    
                    grasp_motion_data['rh2obj_gt'].append(to_cpu(rh2obj_gt['deltas']))
                    grasp_motion_data['rh2obj_gt_ids'].append(to_cpu(rh2obj_gt['closest_ids']))

                    


                    rh2obj_h = self.bps_torch.encode(x= verts_obj,
                                                     feature_type=bps_type_all,
                                                     custom_basis=verts_sbj_h[:, verts_ids[rh_ids_sampled]])

                    grasp_motion_data['rh2obj_h'].append(to_cpu(rh2obj_h['deltas']))
                    grasp_motion_data['rh2obj_h_ids'].append(to_cpu(rh2obj_h['closest_ids']))

                    rh2obj_h_vel = loc2vel(rh2obj_h['dists'],cfg.fps)

                    grasp_motion_data['rh2obj_h_vel'].append(to_cpu(rh2obj_h_vel))

                    rh2obj_p = self.bps_torch.encode(x= verts_obj,
                                                     feature_type=['dists'],
                                                     custom_basis=verts_sbj_p[:, verts_ids[rh_ids_sampled]])['dists']

                    grasp_motion_data['rh2obj_p'].append(to_cpu(rh2obj_p))


                    rh2obj_p_arms = self.bps_torch.encode(x= verts_obj,
                                                        feature_type=['dists'],
                                                        custom_basis=verts_sbj_p_arms[:, verts_ids[rh_ids_sampled]])['dists']
                    
                    grasp_motion_data['rh2obj_p_arms'].append(to_cpu(rh2obj_p_arms))

                    rh2obj_p_arms_vel = loc2vel(rh2obj_p_arms,cfg.fps)

                    grasp_motion_data['rh2obj_p_arms_vel'].append(to_cpu(rh2obj_p_arms_vel))

                    ######################

                    lh2obj_gt = self.bps_torch.encode(x= verts_obj,
                                                     feature_type=bps_type_all,
                                                     custom_basis=verts_sbj[:, verts_ids[lh_ids_sampled]].clone())

                    grasp_motion_data['lh2obj_gt'].append(to_cpu(lh2obj_gt['deltas']))
                    grasp_motion_data['lh2obj_gt_ids'].append(to_cpu(lh2obj_gt['closest_ids']))

                    ###############  From XRZ ###############
                    
                    hoi_obj_data['lh_obj_dists'].append(to_cpu(lh2obj_gt['dists']))
                    hoi_obj_data['rh_obj_dists'].append(to_cpu(rh2obj_gt['dists']))

                    ###############  From XRZ ###############

                    lh2obj_h = self.bps_torch.encode(x= verts_obj,
                                                     feature_type=bps_type_all,
                                                     custom_basis=verts_sbj_h[:, verts_ids[lh_ids_sampled]].clone())
                    
                    grasp_motion_data['lh2obj_h'].append(to_cpu(lh2obj_h['deltas']))
                    grasp_motion_data['lh2obj_h_ids'].append(to_cpu(lh2obj_h['closest_ids']))

                    
                    lh2obj_h_vel = loc2vel(lh2obj_h['dists'],cfg.fps)
                    grasp_motion_data['lh2obj_h_vel'].append(to_cpu(lh2obj_h_vel))

                    lh2obj_p = self.bps_torch.encode(x= verts_obj,
                                                     feature_type=['dists'],
                                                     custom_basis=verts_sbj_p[:, verts_ids[lh_ids_sampled]].clone())['dists']
                   


                    grasp_motion_data['lh2obj_p'].append(to_cpu(lh2obj_p))

                    lh2obj_p_arms = self.bps_torch.encode(x= verts_obj,
                                                        feature_type=['dists'],
                                                        custom_basis=verts_sbj_p_arms[:, verts_ids[lh_ids_sampled]].clone())['dists']
                    
                    grasp_motion_data['lh2obj_p_arms'].append(to_cpu(lh2obj_p_arms))

                    lh2obj_p_arms_vel = loc2vel(lh2obj_p_arms,cfg.fps)

                    grasp_motion_data['lh2obj_p_arms_vel'].append(to_cpu(lh2obj_p_arms_vel))
                    

                    contact_array = seq_data.contact.object[frame_mask][1:]
                    rh_contact = np.isin(contact_array, cfg.include_joints).any(axis=-1)
                    lh_contact = np.isin(contact_array, cfg.exclude_joints).any(axis=-1)

                    ######################  From XRZ #####################

                    rh_windows = find_windows(rh_contact)
                    lh_windows = find_windows(lh_contact)

                    num_frames = rh_contact.shape[0]
                    rh_labels = torch.zeros(num_frames, 1)
                    lh_labels = torch.zeros(num_frames, 1)

                    last_end = 0
                    for start, end in rh_windows:
                        gs = (last_end + start) // 2
                        rh_labels[gs:end, :] = 1
                        last_end = end

                    last_end = 0
                    for start, end in lh_windows:
                        gs = (last_end + start) // 2
                        lh_labels[gs:end, :] = 1
                        last_end = end

                    hoi_obj_data['rh_labels'].append(to_cpu(rh_labels))
                    hoi_obj_data['lh_labels'].append(to_cpu(lh_labels))

                    global_rotation_matrot = local2global_pose(
                        motion_sbj['fullpose_rotmat'],
                        self.kintree
                    )

                    lh_transl = joints_sbj_fhand[1:, 28, :]
                    rh_transl = joints_sbj_fhand[1:, 43, :]

                    lh_orient_mat = global_rotation_matrot[:, 28, ...]
                    rh_orient_mat = global_rotation_matrot[:, 43, ...]

                    lh_obj_rep = self.get_obj2hand(
                        verts_obj,
                        lh_transl.squeeze(1),
                        lh_orient_mat.squeeze(1)
                        
                    )

                    rh_obj_rep = self.get_obj2hand(
                        verts_obj,
                        rh_transl.squeeze(1),
                        rh_orient_mat.squeeze(1)
                    )

                    hoi_obj_data['lh_obj_rep'].append(to_cpu(lh_obj_rep))
                    hoi_obj_data['rh_obj_rep'].append(to_cpu(rh_obj_rep))

                    ######################  From XRZ #####################

                    grasp_motion_data['rh_contact'].append(to_tensor(rh_contact, dtype=torch.float32))
                    grasp_motion_data['lh_contact'].append(to_tensor(lh_contact, dtype=torch.float32))

                    
                    trans_rh2obj = - joints_sbj[:,43:44]
                    R_rh_glob = smplx_loc2glob(motion_sbj['fullpose_rotmat'])[:, 21]
                    rot_rh2obj = R_rh_glob.transpose(1,2)

                    # grasp_motion_data['rh2obj_gt'].append(to_cpu(rh2obj_gt))
                    grasp_motion_data['trans_rh2obj'].append(to_cpu(trans_rh2obj))
                    grasp_motion_data['rot_rh2obj'].append(to_cpu(rot_rh2obj))


                    trans_lh2obj = - joints_sbj[:,28:29]
                    R_lh_glob = smplx_loc2glob(motion_sbj['fullpose_rotmat'])[:, 20]
                    rot_lh2obj = R_lh_glob.transpose(1,2)

                    grasp_motion_data['trans_lh2obj'].append(to_cpu(trans_lh2obj))
                    grasp_motion_data['rot_lh2obj'].append(to_cpu(rot_lh2obj))


                    frame_names.extend(['%s_%s' % (sequence.split('.')[0], fId) for fId in range(T-1)])

                    grasp_motion_data['frame_ids'].extend(to_tensor([fId for fId in range(T-1)], dtype=torch.int16))
                    grasp_motion_data['frame_ends'].extend(to_tensor([T-2 for _ in range(T-1)], dtype=torch.int16))

            self.logger('Processing for %s split finished' % split)
            self.logger('Total number of frames for %s split is:%d' % (split, len(frame_names)))

            # out_data = [grasp_motion_data]
            # out_data_name = ['grasp_motion_data']

            ############### From XRZ ###############

            out_data = [grasp_motion_data, normal_motion_data, hoi_obj_data]
            out_data_name = ['grasp_motion_data', 'normal_motion_data', 'hoi_obj_data']


            ############### From XRZ ###############


            for idx, _ in enumerate(out_data):
                # data = np2torch(data)
                data_name = out_data_name[idx]
                out_data[idx] = torch2np(out_data[idx])
                outfname = makepath(os.path.join(self.out_path, split, '%s.npy' % data_name), isfile=True)
                pickle.dump(out_data[idx], open(outfname, 'wb'), protocol=4)

            np.savez(os.path.join(self.out_path, split, 'frame_names.npz'), frame_names=frame_names)

            np.save(os.path.join(self.out_path, 'obj_info.npy'), self.obj_info)
            np.save(os.path.join(self.out_path, 'sbj_info.npy'), self.sbj_info)

        # print('hi')
        
    
    def process_sequences(self):

        for sequence in self.all_seqs:
            subject_id = sequence.split('/')[-2]
            action_name = os.path.basename(sequence)
            object_name = action_name.split('_')[0]

            # filter data based on the motion intent

            if 'all' in self.intent:
                pass
            elif 'use' in self.intent and any(intnt in action_name for intnt in INTENTS[:3]):
                continue
            elif all([item not in action_name for item in self.intent]):
                continue

            # group motion sequences based on objects
            if object_name not in self.obj_based_seqs:
                self.obj_based_seqs[object_name] = [sequence]
            else:
                self.obj_based_seqs[object_name].append(sequence)

            # group motion sequences based on subjects
            if subject_id not in self.sbj_based_seqs:
                self.sbj_based_seqs[subject_id] = [sequence]
            else:
                self.sbj_based_seqs[subject_id].append(sequence)

            # split train, val, and test sequences
            self.selected_seqs.append(sequence)
            if object_name in self.splits['test']:
                self.split_seqs['test'].append(sequence)
            elif object_name in self.splits['val']:
                self.split_seqs['val'].append(sequence)
            else:
                self.split_seqs['train'].append(sequence)
                if object_name not in self.splits['train']:
                    self.splits['train'].append(object_name)


    # def load_obj_verts(self, obj_name, seq_data, n_verts_sample=2048):

    #     mesh_path = os.path.join(self.grab_path, seq_data.object.object_mesh)
    #     if obj_name not in self.obj_info:
    #         np.random.seed(100)
    #         # obj_mesh = Mesh(filename=mesh_path)
    #         obj_mesh = o3d.io.read_triangle_mesh(mesh_path)
    #         verts_obj = np.array(obj_mesh.vertices)
    #         faces_obj = np.array(obj_mesh.triangles)

    #         if verts_obj.shape[0] > n_verts_sample:
    #             verts_sample_id = np.random.choice(verts_obj.shape[0], n_verts_sample, replace=False)
    #         else:
    #             verts_sample_id = np.arange(verts_obj.shape[0])

    #         verts_sampled = verts_obj[verts_sample_id]
    #         self.obj_info[obj_name] = {'verts': verts_obj,
    #                                    'faces': faces_obj,
    #                                    'verts_sample_id': verts_sample_id,
    #                                    'verts_sample': verts_sampled,
    #                                    'obj_mesh_file': mesh_path}

    #     return self.obj_info[obj_name]

    ####################### From XRZ ###############
    def get_obj2hand(self, obj_verts, hand_transl, hand_rot):
        hand_quat = matrix_to_quaternion(hand_rot)
        obj_v_cart = g2l_cartesian(obj_verts, hand_transl, hand_quat)
        obj_v_sph = cartesian2spherical(obj_v_cart)
        return obj_v_sph


    def load_obj_verts(self, obj_name, seq_data, n_verts_sample=1024):
        mesh_path = os.path.join(self.grab_path, seq_data.object.object_mesh)
        if obj_name not in self.obj_info:
            obj_mesh = o3d.io.read_triangle_mesh(mesh_path)

            v_dim = len(obj_mesh.vertices)

            while v_dim < n_verts_sample:
                obj_mesh = obj_mesh.subdivide_midpoint(number_of_iterations=1)
                v_dim = len(obj_mesh.vertices)

            verts_obj = np.array(obj_mesh.vertices)
            faces_obj = np.array(obj_mesh.triangles)

            obj_bps_torch = bps_torch(n_bps_points=n_verts_sample)
            bps_sample = obj_bps_torch.encode(
                x=to_tensor(verts_obj),
                feature_type=['closest']
            )
            # print('BPS Device: ', bps_sample['closest'].device)
            self.obj_info[obj_name] = {'verts': verts_obj,
                                        'faces': faces_obj,
                                        'verts_sample_id': bps_sample['closest_ids'].cpu(),
                                        'verts_sample': bps_sample['closest'].cpu(),
                                        'obj_mesh_file': mesh_path}

        return self.obj_info[obj_name]
            



    def load_sbj_verts(self, sbj_id, seq_data):

        mesh_path = os.path.join(self.grab_path, seq_data.body.vtemp)
        betas_path = mesh_path.replace('.ply', '_betas.npy')
        if sbj_id in self.sbj_info:
            sbj_vtemp = self.sbj_info[sbj_id]['vtemp']
        else:
            # sbj_vtemp = np.array(Mesh(filename=mesh_path).v)
            sbj_vtemp = np.array(o3d.io.read_triangle_mesh(mesh_path).vertices)
            sbj_betas = np.load(betas_path)
            self.sbj_info[sbj_id] = {'vtemp': sbj_vtemp,
                                     'gender': seq_data.gender,
                                     'betas': sbj_betas}
        return sbj_vtemp

####################### From XRZ ###############
def find_windows(contact_labels, window_thres=10):
    assert isinstance(contact_labels, np.ndarray)
    num_frames = contact_labels.shape[0]

    windows = []
    # contacts = []
    window_s = None
    window_e = None
    prev_in_contact = False
    in_contact = False

    # 逐帧遍历
    for frame_id in range(num_frames):
        in_contact = contact_labels[frame_id] 
        if not prev_in_contact and in_contact:
            # 如果前一帧没有接触，当前帧接触，标记为窗口开始帧
            window_s = frame_id
            prev_in_contact = True

        elif prev_in_contact and in_contact:
            # 如果前一帧接触，当前帧接触，继续延长窗口
            continue

        elif not prev_in_contact and not in_contact:
            # 如果前一帧没有接触，当前帧没有接触，继续等待
            continue

        elif prev_in_contact and not in_contact:
            # 如果前一帧接触，当前帧没有接触，窗口结束
            window_e = frame_id - 1 # 窗口结束帧为上一帧
            prev_in_contact = False # 重置状态

            # 如果窗口长度小于阈值 `window_thres`，则跳过
            if window_e - window_s + 1 < window_thres:
                continue

            windows.append([window_s, window_e])

        else:
            assert False, 'Unreachable code'
    if prev_in_contact:
        window_e = num_frames - 1
        if window_e - window_s + 1 >= window_thres:
            windows.append([window_s, window_e])

    # 验证每个窗口是否具有连续的接触且为最大窗口
    for window in windows:
        line = contact_labels[window[0] - 1 : window[1] + 2].astype(int)

        assert not contact_labels[window[0] - 1]
        assert not contact_labels[window[1] + 1]

        assert sum(line) == len(line) - 2

    return windows


def full2bone(pose, trans, expr):
    global_orient = pose[:, :3]
    body_pose = pose[:, 3:66]
    jaw_pose = pose[:, 66:69]
    leye_pose = pose[:, 69:72]
    reye_pose = pose[:, 72:75]
    left_hand_pose = pose[:, 75:120]
    right_hand_pose = pose[:, 120:]

    body_parms = {'global_orient': global_orient, 'body_pose': body_pose,
                  'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose,
                  'left_hand_pose': left_hand_pose, 'right_hand_pose': right_hand_pose,
                  'transl': trans, 'expression': expr}
    return body_parms


def glob2rel(motion_sbj, motion_obj, R, root_offset, wind, past, rel_trans=None):
    fpose_sbj_rotmat = aa2rotmat(motion_sbj['fullpose'])
    global_orient_sbj_rel = rotmul(R, fpose_sbj_rotmat[:, 0])
    fpose_sbj_rotmat[:, 0] = global_orient_sbj_rel

    trans_sbj_rel = rotate((motion_sbj['transl'] + root_offset), R) - root_offset
    trans_obj_rel = rotate(motion_obj['transl'], R)

    global_orient_obj_rotmat = aa2rotmat(motion_obj['global_orient'])
    global_orient_obj_rel = rotmul(global_orient_obj_rotmat, R.transpose(1, 2))

    if rel_trans is None:
        rel_trans = trans_sbj_rel.clone().reshape(wind, -1)[past:past + 1]
        rel_trans[:, 1] -= rel_trans[:, 1]

    motion_sbj['transl'] = to_tensor(trans_sbj_rel) - rel_trans
    motion_sbj['global_orient'] = rotmat2aa(to_tensor(global_orient_sbj_rel).squeeze()).squeeze()
    motion_sbj['global_orient_rotmat'] = to_tensor(global_orient_sbj_rel)
    motion_sbj['fullpose'][:, :3] = motion_sbj['global_orient']
    motion_sbj['fullpose_rotmat'] = fpose_sbj_rotmat

    motion_obj['transl'] = to_tensor(trans_obj_rel) - rel_trans
    motion_obj['global_orient'] = rotmat2aa(to_tensor(global_orient_obj_rel).squeeze()).squeeze()
    motion_obj['global_orient_rotmat'] = to_tensor(global_orient_obj_rel)

    return motion_sbj, motion_obj, rel_trans


def rel2glob(motion_sbj, motion_obj, R, root_offset, T, past, future, rel_trans=None):
    wind = past + future + 1

    fpose_sbj_rotmat = aa2rotmat(motion_sbj['fullpose'])
    global_orient_sbj_rel = rotmul(R, fpose_sbj_rotmat[:, 0])
    fpose_sbj_rotmat[:, 0] = global_orient_sbj_rel

    trans_sbj_rel = rotate((motion_sbj['transl'] + root_offset), R) - root_offset
    trans_obj_rel = rotate(motion_obj['transl'], R)

    global_orient_obj_rotmat = aa2rotmat(motion_obj['global_orient'])
    global_orient_obj_rel = rotmul(global_orient_obj_rotmat, R.transpose(1, 2))

    if rel_trans is None:
        rel_trans = trans_sbj_rel.reshape(T, wind + 1, -1)
        rel_trans = rel_trans[:, past:past + 1].repeat(1, wind + 1, 1).reshape(-1, 3)

    motion_sbj['transl'] = to_tensor(trans_sbj_rel) - rel_trans
    motion_sbj['global_orient'] = rotmat2aa(to_tensor(global_orient_sbj_rel).squeeze()).squeeze()
    motion_sbj['global_orient_rotmat'] = to_tensor(global_orient_sbj_rel)
    motion_sbj['fullpose'][:, :3] = motion_sbj['global_orient']
    motion_sbj['fullpose_rotmat'] = fpose_sbj_rotmat

    motion_obj['transl'] = to_tensor(trans_obj_rel) - rel_trans
    motion_obj['global_orient'] = rotmat2aa(to_tensor(global_orient_obj_rel).squeeze()).squeeze()
    motion_obj['global_orient_rotmat'] = to_tensor(global_orient_obj_rel)

    return motion_sbj, motion_obj, rel_trans


def loc2vel(loc, fps):
    B = loc.shape[0]
    idxs = [0] + list(range(B - 1))
    vel = (loc[1:] - loc[:-1]) / (1 / float(fps))
    return vel[idxs]

def loc2vel_rot(loc, fps):
    B = loc.shape[0]
    idxs = [0] + list(range(B - 1))
    vel = torch.matmul(loc[1:],loc[:-1].transpose(1,2)) / (1 / float(fps))
    return vel[idxs]

# import trimesh
# def simplify_mesh(mesh=None, v=None, f=None, n_faces=1000, vc=name_to_rgb['pink']):

#     if mesh is None:
#         mesh_tri = trimesh.Trimesh(vertices=v, faces=f).simplify_quadratic_decimation(n_faces)
#     else:
#         mesh_tri = trimesh.Trimesh(vertices=mesh.v, faces=mesh.f).simplify_quadratic_decimation(n_faces)
#     return Mesh(v=mesh_tri.vertices, f=mesh_tri.faces, vc=vc)

def g2l_cartesian(global_verts, transl, rot_quat):
    ''' 给定3d全局笛卡尔坐标系中一个点的坐标与旋转, 以及想要转换的点的集合, 
    计算出每个点在以给定点为原点的坐标系中的局部坐标

            Parameters:
            ----------
            global_verts: torch.Tensor, (num_frames, obj_v_dim, 3)
                物体顶点全局坐标
            transl: torch.Tensor, (num_frames, 3)
                手部坐标系原点
            rot_quat: torch.Tensor, (num_frames, 4)
                手部坐标系的旋转四元数

            Returns:
            ----------
            local_verts: torch.Tensor, (num_frames, obj_v_dim, 3)
    '''
    # 平移
    transl = transl.unsqueeze(1).repeat(1, global_verts.shape[1], 1)
    local_verts = global_verts - transl

    # 旋转
    local_verts_rotated = quaternion_apply(
        rot_quat.unsqueeze(1).repeat(1, global_verts.shape[1], 1),
        local_verts
        )
    
    return local_verts_rotated


# 写这个函数，原意是为了将物体顶点坐标转换为手部球坐标系的坐标
def cartesian2spherical(points):
    ''' 给定3d笛卡尔坐标系中的一组点, 计算出它们在球面坐标系中的坐标

            Parameters:
            ----------
            points: torch.Tensor, (num_frames, obj_v_dim, 3)
                物体顶点坐标

            Returns:
            ----------
            spherical_coords: torch.Tensor, (num_frames, obj_v_dim, 3)
                物体顶点在球面坐标系中的坐标
    '''
    r = torch.sqrt(torch.sum(points ** 2, dim=-1))
    theta = torch.atan2(points[..., 1], points[..., 0])
    phi = torch.acos(torch.clamp(points[..., 2] / r, -1.0, 1.0))
    spherical_coords = torch.stack([r, theta, phi], dim=-1)
    return spherical_coords

if __name__ == '__main__':
    # import argparse

    parser = argparse.ArgumentParser(description='MNet-dataset')

    parser.add_argument('--grab-path',
                        required=True,
                        type=str,
                        help='The path to the folder that contains GRAB data')

    parser.add_argument('--smplx-path',
                        required=True,
                        type=str,
                        help='The path to the folder containing SMPL-X model downloaded from the website')
    
    parser.add_argument('--out-path',
                        required=True,
                        type=str,
                        help='The path to the folder that the processed data will be saved')

    cmd_args = parser.parse_args()

    grab_path = cmd_args.grab_path
    model_path = cmd_args.smplx_path

    # exp_id = 'V05_multi_both_12_with_ids_arm_ICCV'
    exp_id = 'Data_GRIP_00'


    out_path = os.path.join(cmd_args.out_path, exp_id)
    makepath(out_path)

    # split the dataset based on the objects
    grab_splits = {'test': ['mug', 'camera', 'binoculars', 'apple', 'toothpaste'],
                   'val': ['fryingpan', 'toothbrush', 'elephant', 'hand'],
                   'train': []}

    cfg = {

        'intent': ['all'],  # from 'all', 'use' , 'pass', 'lift' , 'offhand'

        'save_contact': False,  # if True, will add the contact info to the saved data
        # motion fps (default is 120.)
        'fps': 30.,
        'past': 10,  # number of past frames to include
        'future': 10,  # number of future frames to include
        ### splits
        'splits': grab_splits,

        ###IO path
        'grab_path': grab_path,
        'out_path': out_path,

        ### number of vertices samples for each object
        'n_verts_sample': 1024,

        ### body and hand model path
        'model_path': model_path,

        ### include/exclude joints
        'include_joints': list(range(41, 53)),
        # 'required_joints' : [16],  # mouth
        'required_joints': list(range(53, 56)),  # thumb
        'exclude_joints': list(range(26, 41)),

        ### bps info
        'r_obj': .15,
        'n_obj': 1024,

        'r_sbj': 1.5,
        'n_sbj': 1024,
        'g_size': 20,
        'h_sbj': 2.,

        'r_rh': .2,
        'n_rh': 1024,

        'r_lh': .2,
        'n_lh': 1024,

        'r_hd': .15,
        'n_hd': 1024,

        ### interpolaton params
        'interp_frames': 60,
        'fix_length': False,

    }

    cwd = os.getcwd()
    default_cfg_path = os.path.join(cwd, '../configs/grab_preprocessing_cfg.yaml')
    cfg = Config(default_cfg_path=default_cfg_path, **cfg)
    makepath(cfg.out_path)
    cfg.write_cfg(write_path=cfg.out_path + '/grab_preprocessing_cfg.yaml')

    log_dir = os.path.join(cfg.out_path, 'grab_processing.log')
    logger = makelogger(log_dir=log_dir, mode='a').info

    MNetDataSet(cfg, logger)
