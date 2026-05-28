import os
import cv2
import glob
import Imath
import shutil
import OpenEXR
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

def exr2hdr(exrpath):
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    CNum = len(File.header()['channels'].keys())
    if (CNum > 1):
    	Channels = ['R', 'G', 'B']
    	CNum = 3
    else:
    	Channels = ['G']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    Pixels = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in Channels]
    hdr = np.zeros((Size[1],Size[0],CNum),dtype=np.float32)
    if (CNum == 1):
        hdr[:,:,0] = np.reshape(Pixels[0],(Size[1],Size[0]))
    else:
	    hdr[:,:,0] = np.reshape(Pixels[0],(Size[1],Size[0]))
	    hdr[:,:,1] = np.reshape(Pixels[1],(Size[1],Size[0]))
	    hdr[:,:,2] = np.reshape(Pixels[2],(Size[1],Size[0]))
    return hdr

def load_exr(filename):
	hdr = exr2hdr(filename)
	h, w, c = hdr.shape
	if c == 1:
		hdr = np.squeeze(hdr)
	return hdr

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    return parser.parse_args()

def convert_to_depth(disparity):
    baseline = 0.1
    focal_length = 480

    depth = baseline * focal_length / disparity

    return depth

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    scenes = os.listdir(data_path)

    # image and depth
    for scene in ['Home', 'Office', 'Restaurant', 'Store']:
        sub_scenes = os.listdir(os.path.join(data_path, scene))
        for sub_scene in tqdm(sub_scenes):
            files = glob.glob(os.path.join(data_path, scene, sub_scene, 'd_*'))
            for file in tqdm(files):
                disparity = load_exr(file)
                # convert disparity to depth
                depth = convert_to_depth(disparity)

                depth_output_path = os.path.join(output_path, scene, sub_scene, 'depth')
                os.makedirs(depth_output_path, exist_ok=True)
                depth_id = file.split('/')[-1].split('.')[0]
                depth_output_file = os.path.join(depth_output_path, depth_id + '.npy')
                np.save(depth_output_file, depth)

                rgb_id = depth_id.replace('d_', 'l_') + '.png'
                rgb_output_path = os.path.join(output_path, scene, sub_scene, 'rgb')
                os.makedirs(rgb_output_path, exist_ok=True)
                rgb_source_file = os.path.join(data_path, scene, sub_scene, rgb_id)
                rgb_output_file = os.path.join(rgb_output_path, rgb_id)
                shutil.copy(rgb_source_file, rgb_output_file)

if __name__ == "__main__":
    main()
