'''
This script generates discs labels using SPINEPS' vertebrae segmentation

Author: Nathan Molinier
'''
import os
import cc3d
import argparse
import numpy as np
from image import Image
from pathlib import Path

DISCS_MAP = {2:1, 102: 3, 103: 4, 104: 5, 
             105: 6, 106: 7, 107: 8, 
             108: 9, 109: 10, 110: 11, 
             111: 12, 112: 13, 113: 14, 
             114: 15, 115: 16, 116: 17, 
             117: 18, 118: 19, 119: 20, 
             120: 21, 121: 22, 122: 23, 
             123: 24, 124: 25}


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 database format.')
    parser.add_argument('--path-vert', type=str, required=True, help='Path to the SPINEPS vertebrae labels. Example: "/<path_to_BIDS_data>/derivatives/labels/sub-amuALT/anat/sub-amuALT_T2w_label-vert_dseg.nii.gz" (Required)')
    parser.add_argument('--path-out', type=str, default='', help='Output path of the discs label. Example: "/<path_to_BIDS_data>/derivatives/labels/sub-amuALT/anat/sub-amuALT_T2w_label-discs_dlabel.nii.gz". By default, the structure "_label-discs_dlabel" will be used.')
    return parser


def main():
    # Load parser
    parser = get_parser()
    args = parser.parse_args()

    # Fetch paths
    path_in = os.path.abspath(args.path_vert)
    path_out = os.path.abspath(args.path_out) if args.path_out else default_name_discs(path_in)

    # Check if output folder exists
    if not os.path.exists(os.path.dirname(path_out)):
        os.makedirs(os.path.dirname(path_out))

    # Extract discs labels
    vert_image = Image(path_in)
    print('-'*80)
    print(f"Creating discs label using SPINEPS prediction: {path_in}")
    print('-'*80)
    discs_nii_clean = extract_discs_label(vert_image, mapping=DISCS_MAP)

    # Save discs labels
    discs_nii_clean.save(path_out)
    print('-'*80)
    print(f'Discs label: {path_out} was created.')
    print('-'*80)


def default_name_discs(path_in, suffix="_label-discs_dlabel"):
    '''
    Generate default discs label name
    '''
    # Fetch suffixes
    path_obj = Path(path_in)
    ext = ''.join(path_obj.suffixes)

    # Check if the entity label is already present in the filename
    fname = os.path.basename(path_in).replace(ext,'')
    ofolder = os.path.dirname(path_in)
    if '_label-vert' in fname:
        fname = fname.split('_label-vert')[0]
    
    # Reconstruct out path
    path_out = os.path.join(ofolder, fname+suffix+ext)
    return path_out



def extract_discs_label(label, mapping):
    # Store input orientation
    orig_orientation = label.orientation

    # Use RSP orientation
    label.change_orientation('RSP')

    # Extract only discs segmentations based on mapping
    data = label.data
    data_discs_seg = np.zeros_like(data)
    for seg_value, discs_value in mapping.items():
        data_discs_seg[np.where(data==seg_value)] = discs_value
    
    # Deal with disc 1 obtained with the first vertebrae (Highest vertical coordinate)
    if 1 in data_discs_seg: 
        # If the first vertebrae is present identify label disc 1 at the top
        vert1_seg = np.array(np.where(data_discs_seg==1))
        disc1_idx = np.argmin(vert1_seg[1]) # find min on the S-I axis
        disc1_coord = vert1_seg[:,disc1_idx]
        data_discs_seg[np.where(data_discs_seg==1)] = 0
        data_discs_seg[disc1_coord[0], disc1_coord[1], disc1_coord[2]] = 1
    
    ## Identify the posterior tip of the disc
    # Extract the center of mass of every discs segmentation to create discs labels
    discs_centroids, discs_bb = extract_centroids_3D(data_discs_seg) # Centroids are sorted based on the vertical axis

    # Generate a centerline between the discs by doing linear interpolation
    yvals = np.linspace(discs_centroids[0, 1], discs_centroids[-1, 1], round(8*len(discs_centroids))) # TODO: Should we calculate the number of dots based on the resolution ?
    xvals = np.interp(yvals, discs_centroids[:,1], discs_centroids[:,0])
    zvals = np.interp(yvals, discs_centroids[:,1], discs_centroids[:,2])
    centerline = np.concatenate((np.expand_dims(xvals, axis=1), np.expand_dims(yvals, axis=1), np.expand_dims(zvals, axis=1)), axis=1)

    # Shift the centerline to the posterior direction until there is no intersection with the discs segmentations
    min_seg_AP = np.min(np.where(data_discs_seg>0)[2]) # Find the min coordinate of the discs segmentation on the A-P axis
    max_centroid_AP = np.max(discs_centroids[:,2])
    offset = 5
    shift = (max_centroid_AP - min_seg_AP + offset) if min_seg_AP >= offset else (max_centroid_AP - min_seg_AP)
    centerline_shifted = np.copy(centerline)
    centerline_shifted[:,2] = centerline_shifted[:,2] - shift

    # For each segmented disc, identify the closest voxel to this shifted centerline
    discs_list = closest_point_seg_to_line(data_discs_seg, centerline_shifted, discs_bb)

    # Add disc 2 between disc 1 and 3
    if 1 and 3 in discs_list[:,-1]:
        disc1_coord = discs_list[discs_list[:,-1]==1]
        disc2_coord = discs_list[discs_list[:,-1]==3]
        disc2_coord[0,1] = (disc2_coord[0,1] + disc1_coord[0,1])//2
        disc2_coord[0,-1] = 2
        discs_list = np.insert(discs_list, 1, disc2_coord, axis=0)

    # Create output Image
    data_discs = np.zeros_like(data)
    for x, y, z, v in discs_list:
        data_discs[x, y, z] = v
    label.data = data_discs
    return label.change_orientation(orig_orientation)


def extract_centroids_3D(arr):
    '''
    Extract centroids and bouding boxes from a 3D numpy array
    :param arr: 3D numpy array
    '''
    stats = cc3d.statistics(cc3d.connected_components(arr))
    centroids = stats['centroids'][1:] # Remove backgroud <0>
    bounding_boxes = stats['bounding_boxes'][1:]
    sort_args = np.argsort(centroids[:,1]) # Sort according to the vertical axis because RSP orientation
    centroids_sorted = centroids[sort_args]
    bb_sorted = np.array(bounding_boxes)[sort_args]
    return centroids_sorted.astype(int), bb_sorted


def project_point_on_line(point, line):
    """
    Project the input point on the referenced line by finding the minimal distance

    :param point: coordinates of a point and its value: point = numpy.array([x y z])
    :param line: list of points coordinates which composes the line
    :returns: closest coordinate to the referenced point on the line: projected_point = numpy.array([X Y Z])
    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox
    """
    # Calculate distances between the referenced point and the line then keep the closest point
    dist = np.sum((line - point) ** 2, axis=1)

    return line[np.argmin(dist)], np.min(dist)

def closest_point_seg_to_line(discs_seg, centerline, bounding_boxes):
    """
    """
    discs_list = []
    for x, y, z in bounding_boxes:
        zer = np.zeros_like(discs_seg)
        zer[x, y, z] = discs_seg[x, y, z] # isolate disc
        # Loop on all the pixels of the segmentation
        min_dist = np.inf
        nonzero = np.where(zer>0)
        for u, v, w in zip(nonzero[0],nonzero[1],nonzero[2]):
            point, dist = project_point_on_line(np.array([u, v, w]), centerline)
            if dist < min_dist:
                min_dist = dist
                min_point = np.array([u, v, w, discs_seg[u, v, w]])
        discs_list.append(min_point)
    return np.array(discs_list)


if __name__=='__main__':
    main()