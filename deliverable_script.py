import os
import math
import colorsys
import numpy as np
from shutil import copyfile
import cv2

shirts_dataset_folder = 'shirt_dataset'
imgs_path_result = 'resulting_blue'

saturation_thr = 30


def get_samples_in_feature_space(imgs_path):
    hsv_samples = []
    no_regions = []
    fld_len = len(os.listdir(imgs_path))

    for iif, imgfile in enumerate(os.listdir(imgs_path)):
        img = cv2.imread(os.path.join(imgs_path, imgfile))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        threshhsv = hsv[..., 1] > saturation_thr
        thresh = (threshhsv * 255).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(img, markers)

        img[markers == -1] = [255, 0, 0]

        msk_x1 = int(img.shape[0] * 0.1)
        msk_x2 = int(img.shape[0] * 0.9)
        msk_y1 = int(img.shape[1] * 0.1)
        msk_y2 = int(img.shape[1] * 0.9)

        msk_center = np.zeros(markers.shape)
        msk_center[msk_x1:msk_x2, msk_y1:msk_y2] = 1
        markers_centered = msk_center * markers

        bin_total_msk = np.zeros(markers_centered.shape)

        unv = np.unique(markers_centered, return_counts=True)[0]
        occurences = np.unique(markers_centered, return_counts=True)[1]

        for val, nocc in zip(unv, occurences):
            if nocc > bin_total_msk.size * 0.01 and val not in (-1, 0, 1):
                bin_total_msk[markers_centered == val] = 1

        if bin_total_msk.sum() < bin_total_msk.size * 0.1:
            bin_total_msk = np.zeros(markers_centered.shape)
        else:
            msk_bbox = np.array([np.array(np.where(bin_total_msk == 1)).min(axis=1),
                                 np.array(np.where(bin_total_msk == 1)).max(
                                     axis=1)]) if bin_total_msk.any() else np.zeros((2, 2))
            msk_bbox_size = (msk_bbox[1, 0] - msk_bbox[0, 0], msk_bbox[1, 1] - msk_bbox[0, 1])
            kern_shape = np.ones((int((msk_bbox_size[0] * 0.05) + 1), int((msk_bbox_size[1] * 0.05) + 1)))
            bin_total_closed = cv2.morphologyEx(bin_total_msk, cv2.MORPH_CLOSE, kern_shape, iterations=2)
            tot_area = bin_total_closed[msk_bbox[0, 0]:msk_bbox[1, 0], msk_bbox[0, 1]:msk_bbox[1, 1]].sum()
            lbmargin = msk_bbox[0, 0] + int(0.1 * msk_bbox_size[0])
            rbmargin = msk_bbox[0, 0] + int(0.9 * msk_bbox_size[0])
            tmargin = msk_bbox[0, 1] + int(0.5 * msk_bbox_size[1])
            area_lb = bin_total_closed[msk_bbox[0, 0]:lbmargin, tmargin:msk_bbox[1, 1]].sum()
            area_rb = bin_total_closed[rbmargin:msk_bbox[1, 0], tmargin:msk_bbox[1, 1]].sum()
            area_lb_pen = area_lb * 2
            area_rb_pen = area_rb * 2
            shape_area = tot_area - area_lb_pen - area_rb_pen
            box_area = (msk_bbox[1, 0] - msk_bbox[0, 0]) * (msk_bbox[1, 1] - msk_bbox[0, 1])
            if (shape_area / box_area) < 0.33:
                bin_total_msk = np.zeros(markers_centered.shape)

        masked_values_centered = img[bin_total_msk == 1]

        if masked_values_centered.size > 0:

            infered_color = masked_values_centered.mean(axis=0)

            infered_color_hsv = colorsys.rgb_to_hsv(*infered_color)

            hsv_samples.append(infered_color_hsv[:2])

        else:

            hsv_samples.append((0, 0))

            no_regions.append(iif)

        print('Done ' + str(iif) + '/' + str(fld_len))

    return hsv_samples, no_regions


imgs_path_pool = os.path.join(shirts_dataset_folder, 'unlabeled')

hsv_samples_pool, no_regions_pool = get_samples_in_feature_space(imgs_path_pool)

samples_positioned_pool = [(hsvs[1] * math.cos(hsvs[0] * 2 * math.pi), hsvs[1] * math.sin(hsvs[0] * 2 * math.pi))
                           for hsvs in hsv_samples_pool]

hsv_samples_pool = [hss for iihss, hss in enumerate(hsv_samples_pool) if iihss not in no_regions_pool]
samples_positioned_pool = [spp for iispp, spp in enumerate(samples_positioned_pool) if iispp not in no_regions_pool]

validimgfiles = [vimg for iipif, vimg in enumerate(os.listdir(imgs_path_pool)) if iipif not in no_regions_pool]

no_regions_pool = []

imgs_path_query = os.path.join('shirt_dataset', 'labeled')

hsv_samples_query, no_regions_query = get_samples_in_feature_space(imgs_path_query)

samples_positioned_query = [(hsvs[1] * math.cos(hsvs[0] * 2 * math.pi), hsvs[1] * math.sin(hsvs[0] * 2 * math.pi))
                           for hsvs in hsv_samples_query]

hsv_samples_query = [hss for iihss, hss in enumerate(hsv_samples_query) if iihss not in no_regions_query]
samples_positioned_query = [spp for iispp, spp in enumerate(samples_positioned_query) if iispp not in no_regions_query]

no_regions_query = []

matching_pool = []
mcolors_pool = []
matching_indices = []

min_hue_q = np.array(hsv_samples_query)[:, 0].min()
max_hue_q = np.array(hsv_samples_query)[:, 0].max()
min_sat_q = np.array(hsv_samples_query)[:, 1].min()

for iih, hsvs in enumerate(hsv_samples_pool):
    if min_hue_q < hsvs[0] < max_hue_q and hsvs[1] > min_sat_q:
        matching_pool.append(samples_positioned_pool[iih])
        mcolors_pool.append(hsv_samples_pool[iih])
        matching_indices.append(iih)

if not os.path.exists(imgs_path_result):
    os.mkdir(imgs_path_result)
elif os.listdir(imgs_path_result):
    print('Directory not empty, adding _new suffix')
    imgs_path_result += '_new'
    os.mkdir(imgs_path_result)

for mind in matching_indices:
    copyfile(os.path.join(imgs_path_pool, validimgfiles[mind]),
             os.path.join(imgs_path_result, validimgfiles[mind]))
