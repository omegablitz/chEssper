import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_adaptive
from skimage.filters import threshold_otsu
from scipy.misc import imresize
from PIL import Image
import sys
import random as r

# @profile
def main():
    color_img_orig = cv2.imread(sys.argv[1])
    color_img = color_img_orig.copy()
    height = color_img.shape[0]
    width = color_img.shape[1]

    # img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    # th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    # ret, thresh = cv2.threshold(th2, 127, 255, 0)

    thresh = cv2.cvtColor(color_img_orig, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.medianBlur(thresh, 5) # TRY THIS
    thresh_unscaled = threshold_adaptive(thresh, 251, offset = 10)
    thresh = thresh_unscaled.astype("uint8") * 255
    # mask = threshold_otsu(thresh)
    # thresh = thresh > mask
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    def hsh(x, y):
        return int(round(x / 100) * 10000 + round(y / 100))
    done = set()
        
    def order_points(pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")
        
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        # return the ordered coordinates
        return rect

    def four_point_transform(image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # return the warped image
        return warped

    pts = []
    box_width = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if 4 == len(approx):
            rect = cv2.minAreaRect(c)
            (x, y), (w, h), theta = rect #cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if w >= width / 40.0 and h >= width / 40.0 and hsh(x, y) not in done and 0.8 <= aspect_ratio <= 1.2:
                # cv2.circle(color_img, (int(x), int(y)), 9, 250)
                box_points_floating = cv2.boxPoints(rect)
                box_points = np.int0(box_points_floating)
                mask = np.zeros((height, width), np.uint8)
                cv2.fillPoly(mask, [box_points], 1)
                # color = sum([int(_) for _ in cv2.mean(color_img, mask)[0:3]]) / 3
                # if color < 130:
                color = sum([int(_) for _ in cv2.mean(thresh, mask)[0:3]]) / 3
                if color < 50:
                    box_width = w
                    boxes.append(box_points)
                    done.add(hsh(x, y))
                    pts += box_points.tolist()
                    cv2.circle(color_img, tuple(box_points[0].tolist()), 3, 250)
    # print boxes
    print len(boxes)
    modified = color_img.copy()
    cv2.fillPoly(modified, boxes, 190)

    avg_x, avg_y = map(lambda x: x / len(pts), reduce(lambda x, y: [x[0] + y[0], x[1] + y[1]], pts))
    top_left = None
    top_right = None
    bottom_left = None
    bottom_right = None
    def dist(pt1, pt2):
        return (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2
    for (x, y) in pts:
        if x < avg_x and y < avg_y:
            if top_left == None or dist((x, y), (avg_x, avg_y)) > dist(top_left, (avg_x, avg_y)):
                top_left = (x, y)
        elif x > avg_x and y < avg_y:
            if top_right == None or dist((x, y), (avg_x, avg_y)) > dist(top_right, (avg_x, avg_y)):
                top_right = (x, y)
        elif x < avg_x and y > avg_y:
            if bottom_left == None or dist((x, y), (avg_x, avg_y)) > dist(bottom_left, (avg_x, avg_y)):
                bottom_left = (x, y)
        elif x > avg_x and y > avg_y:
            if bottom_right == None or dist((x, y), (avg_x, avg_y)) > dist(bottom_right, (avg_x, avg_y)):
                bottom_right = (x, y)


    warped = four_point_transform(color_img_orig, np.array([top_left, top_right, bottom_left, bottom_right]))
    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    mask = threshold_adaptive(warped, 251, offset = 10).astype('uint8')
    classify = np.maximum(mask*255, warped) # image we actually use for classification - grayscale
    warped = mask * 255


    images = [color_img, thresh, modified, warped]
    # images = [color_img_orig, warped]

    for i in images:
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image',i)
        cv2.resizeWindow('image', 600, 600)
        cv2.waitKey(0)

    warped_height = warped.shape[0]
    warped_width = warped.shape[1]


    dark_thresh_col = 100
    dark_thresh_row = 100
    cols = []

    kernel_size = 3
    for i in range(warped_width - kernel_size + 1):
        total = 0
        maxes = np.amin(warped[:, i:i + kernel_size], axis=1)
        color = np.sum(maxes) / warped_height
        if color < dark_thresh_col:
            cols.append((i, color))
    cols.sort(key=lambda x: x[0]) # unnecessary
    col_diffs = []
    for col_idx in range(1, len(cols)):
        diff = cols[col_idx][0] - cols[col_idx-1][0]
        if diff > box_width * 5 / 6.0:
            col_diffs.append(diff)

    def median(lst):
        sortedLst = sorted(lst)
        lstLen = len(lst)
        index = (lstLen - 1) // 2

        if (lstLen % 2):
            return sortedLst[index]
        else:
            return (sortedLst[index] + sortedLst[index + 1])/2.0
    col_spacing = median(col_diffs)

    print maxes.shape, warped_height
    rows = []
    for i in range(warped_height - kernel_size + 1):
        maxes = np.amin(warped[i:i + kernel_size, :], axis=0)
        color = np.sum(maxes) / warped_width
        if color < dark_thresh_row:
            rows.append((i, color))
    print maxes.shape, warped_width
    rows.sort(key=lambda x: x[0])

    vert_splits = [] # Tuples of vertical splits
    vert_start = cols[0][0]
    horiz_splits = [] # Tuples of horiz splits
    horiz_start = rows[0][0]
    for x in cols:
        if x[0] - vert_start > box_width * 3 / 4.0:
            print ((x[0] - vert_start) / float(col_spacing)) # 9 / 7 is to scale black box up
            # num_in_between = round(((x[0] - vert_start) / (box_width * 9 / 7))) # 9 / 7 is to scale black box up
            num_in_between = round((x[0] - vert_start) / float(col_spacing)) # 9 / 7 is to scale black box up
            offset_amount = (x[0] - vert_start) / num_in_between
            for idx in range(int(num_in_between)):
                offset = int(round(offset_amount * idx))
                vert_splits.append((vert_start + offset, vert_start + offset + int(offset_amount)))
            vert_start = x[0]
    for x in rows:
        if x[0] - horiz_start > 10:
            horiz_splits.append((horiz_start, x[0]))
        # x[0] is idx
        horiz_start = x[0]

    print 'vert splits', len(vert_splits)
    print 'horiz splits', len(horiz_splits)
    new = cv2.cvtColor(classify, cv2.COLOR_GRAY2BGR)
    for left, right in vert_splits:
        color = (r.random() * 255, r.random() * 255, r.random() * 255)
        cv2.line(new, (left, 0), (left, warped_height - 1), color, 3)
        cv2.line(new, (right, 0), (right, warped_height - 1), color, 3)
        
    for left, right in horiz_splits:
        color = (r.random() * 255, r.random() * 255, r.random() * 255)
        cv2.line(new, (0, left), (warped_width - 1, left), color, 3)
        cv2.line(new, (0, right), (warped_width - 1, right), color, 3)

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', new)
    # cv2.resizeWindow('image', 600, 600)
    # cv2.waitKey(0)

    new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    def get_box(image, x, y, vert_splits, horiz_splits, crop=0):
        startX, endX = vert_splits[x]
        print startX, endX
        startY, endY = horiz_splits[y]
        startX += 1; startY += 1 # We want elements in BETWEEN
        startX += crop; startY += crop; endX -= crop; endY -= crop # Shave off sides
        return 1 - image[startY:endY, startX:endX]
    def autocrop(image, start=0): # returns None if nothing in box
        ###if start == 0:
        ###    kernel = np.ones((2,2),np.uint8)
        ###    image = cv2.erode(image,kernel,iterations = 2)
        for i in range(4):
            if (start + i) % 4 == 0 and np.mean(image[:, 0]) > 50:
                return autocrop(np.delete(image, (0), axis=1), start=start+1)
            # elif (start + i) % 4 == 1 and np.mean(image[:, -1]) > 50:
            #     return autocrop(np.delete(image, (-1), axis=1), start=start+1)
            elif (start + i) % 4 == 2 and np.mean(image[0]) > 50:
                return autocrop(np.delete(image, (0), axis=0), start=start+1)
            elif (start + i) % 4 == 3 and np.mean(image[-1]) > 75:
                return autocrop(np.delete(image, (-1), axis=0), start=start+1)
        ###kernel = np.ones((2,2),np.uint8)
        ###image = cv2.dilate(image,kernel,iterations = 1)

        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        #cv2.imshow('image', image)
        #cv2.resizeWindow('image', 600, 600)
        #cv2.waitKey(0)

        ret, Ithres = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        Ithres = 255 - Ithres
        im2, contours, hierarchy = cv2.findContours(Ithres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        c_biggest_area = None
        box_w = image.shape[1]
        box_h = image.shape[0]
        box_center = (box_w / 2, box_h / 2)
        for c in contours:
            rect = cv2.minAreaRect(c)
            (x, y), (w, h), theta = rect #cv2.boundingRect(approx)
            if w*2+h*2 < (box_w*2+box_h*2) * 3/4 and dist(box_center, (x, y)) < (box_w/4)**2 + (box_h/4)**2 and (c_biggest_area is None or cv2.contourArea(c) > cv2.contourArea(c_biggest_area)):
            #if (c_biggest_area == None or cv2.contourArea(c) > cv2.contourArea(c_biggest_area)):
                c_biggest_area = c
        if c_biggest_area is None or cv2.contourArea(c_biggest_area) < box_w * box_h / 36:
            return None

        x, y, w, h = cv2.boundingRect(c_biggest_area)
        
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image', image)
        # cv2.resizeWindow('image', 600, 600)
        # cv2.waitKey(0)
        image = image[y: y + h + 1, x: x + w + 1]
        if w > h:
            new_height = int(round(h*20.0/w))
            image = imresize(image, (new_height, 20), mode='L')
            image = np.pad(image, [(4 + (20 - new_height) / 2, 4 + (20 - new_height) / 2 + new_height % 2), (4, 4)], mode='constant', constant_values=0)
        else:
            new_width = int(round(w*20.0/h))
            image = imresize(image, (20, new_width), mode='L')
            image = np.pad(image, [(4, 4), (4 + (20 - new_width) / 2, 4 + (20 - new_width) / 2 + new_width % 2)], mode='constant', constant_values=0)
        return image
    for j in range(25):
        for i in range(10):
            print i, j
            box_0_0 = get_box(new, i, j, vert_splits, horiz_splits)
            box_0_0 = autocrop(box_0_0)
            if box_0_0 is None:
                continue
            #ret, box_0_0 = cv2.threshold(box_0_0,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            #box_0_0 = 255 - imresize(box_0_0, (20, 20), mode='L')
            cv2.imwrite('tmp/' + str(j) + '_' + str(i) + '.png', box_0_0)
            # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # cv2.imshow('image', box_0_0)
            # cv2.resizeWindow('image', 600, 600)
            # cv2.waitKey(0)
    for j in range(25):
        for i in range(10):
            i = i + 10
            print i, j
            box_0_0 = get_box(new, i, j, vert_splits, horiz_splits)
            box_0_0 = autocrop(box_0_0)
            if box_0_0 is None:
                continue
            #ret, box_0_0 = cv2.threshold(box_0_0,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            #box_0_0 = 255 - imresize(box_0_0, (20, 20), mode='L')
            cv2.imwrite('tmp/' + str(j + 25) + '_' + str(i - 10) + '.png', box_0_0)
#            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#            cv2.imshow('image', box_0_0)
#            cv2.resizeWindow('image', 600, 600)
#            cv2.waitKey(0)
main()
