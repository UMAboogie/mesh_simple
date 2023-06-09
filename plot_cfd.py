# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Plots a CFD trajectory rollout."""
import torch

import pickle

from absl import app
from absl import flags
from matplotlib import animation
from matplotlib import tri as mtri
import matplotlib.pyplot as plt

import datetime
import os
import pathlib


FLAGS = flags.FLAGS
# flags.DEFINE_string('rollout_path', 'C:\\Users\\Mark\\iCloudDrive\\master_arbeit\\implementation\\meshgraphnets\\output\\cylinder_flow\\Sun-Oct-24-21-30-58-2021\\rollout\\rollout.pkl', 'Path to rollout pickle file')
flags.DEFINE_string('rollout_dir', None, 'Path to rollout pickle file')
flags.DEFINE_string('gif_dir', None, 'dir name of animation')

def main(unused_argv):
    print("Ploting run")
    with open(FLAGS.rollout_dir + "/rollout.pkl", 'rb') as fp:
        rollout_data = pickle.load(fp)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    skip = 10
    num_steps = rollout_data[0]['gt_velocity'].shape[0]
    num_frames = len(rollout_data) * num_steps // skip

    # compute bounds
    bounds = []
    for trajectory in rollout_data:
        bb_min = trajectory['gt_velocity'].cpu().numpy().min(axis=(0, 1))
        bb_max = trajectory['gt_velocity'].cpu().numpy().max(axis=(0, 1))
        bounds.append((bb_min, bb_max))

    def animate(num):
        step = (num * skip) % num_steps
        traj = (num * skip) // num_steps
        ax.cla()
        ax.set_aspect('equal')
        ax.set_axis_off()
        vmin, vmax = bounds[traj]
        pos = rollout_data[traj]['mesh_pos'][step].to('cpu')
        faces = rollout_data[traj]['faces'][step].to('cpu')
        velocity = rollout_data[traj]['pred_velocity'][step].to('cpu')
        triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
        ax.tripcolor(triang, velocity[:, 0], vmin=vmin[0], vmax=vmax[0])
        ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
        ax.set_title('Trajectory %d Step %d' % (traj, step))
        return fig,
    
    anima = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
    # writervideo = animation.FFMpegWriter(fps=30)
    # anima.save(os.path.join(save_path, 'ani.mp4'), writer=writervideo)
    if FLAGS.gif_dir != None:
        os.makedirs(FLAGS.gif_dir, exist_ok=True)
        dt_now = datetime.datetime.now()
        file_name = dt_now.strftime('%y%m%d%H%M') + '_cfd.gif'
        anima.save(os.path.join(FLAGS.gif_dir, file_name), writer="imagemagick")
    else:
        plt.show(block=True)
    """
  
    t0 = rollout_data[0]['pred_velocity'][0]
    t1 = rollout_data[0]['gt_velocity'][1]

    diff = t1 - t0
    for i in range(len(t1)):
        print(t1[i])
    """


if __name__ == '__main__':
    app.run(main)
