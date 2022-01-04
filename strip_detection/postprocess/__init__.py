import numpy as np
import cv2


class Strip:
    STRIP_THRESHOLD = 0.7
    PAD_THRESHOLD = 0.7
    RATIO_THRESHOLD = 0.6

    def __init__(self, img_path, result, num_pads):
        self.rois = result['rois']
        self.class_ids = result['class_ids']
        self.score = result['scores']
        self.masks = result['masks']
        self.num_pads = num_pads
        self.aspect_ratio = self._aspect_ratio()
        self.img = cv2.imread(img_path)

    def _aspect_ratio(self):
        aspect_ratio = []
        for i in range(len(self.rois)):
            y1, x1, y2, x2 = self.rois[i]
            width = x2 - x1
            height = y2 - y1
            aspect_ratio.append(np.float32(height / width))
        return aspect_ratio

    def _prod(self, array):
        result = array[0]
        for i in range(1, len(array)):
            result = result * array[i]
        return result

    def _arrange(self, matrix):
        result = []
        for i in range(0, len(matrix)):
            mult = self._prod(matrix[i])
            result.append(mult)
        return result

    def _get_strip_pad(self):
        self.strip = []
        self.pad = []
        for i in range(len(self.class_ids)):
            if self.class_ids[i] == 1 and self.score[i] >= self.STRIP_THRESHOLD:
                self.strip.append([self.rois[i], self.masks[i]])
            elif self.class_ids[i] == 2:
                if self.aspect_ratio[i] >= self.RATIO_THRESHOLD and self.score[i] >= self.PAD_THRESHOLD:
                    self.pad.append(self.rois[i])

        if not self.strip:
            raise Exception("Strip is not detected.")
        elif len(self.strip) != 1:
            raise Exception("There must be one strip.")
        elif len(self.pad) != self.num_pads:
            raise Exception(f"{len(self.pad)} pads are detected. There must be {self.num_pads} pads.")
        else:
            return self.strip, self.pad

    def check_pad(self):
        img_copy = self.img.copy()
        strip, pad = self._get_strip_pad()

        for i in range(len(pad)):
            y1, x1, y2, x2 = pad[i]
            x_center = np.int32((x1 + x2) / 2)
            y_center = np.int32((y1 + y2) / 2)
            cv2.circle(img_copy, (x_center, y_center), 15, (0, 0, 0), 1)

        print("Check if circles are at the center of each pad.")
        return img_copy

    def black_bg(self):
        img_copy = self.img.copy()

        strip, pad = self._get_strip_pad()
        strip_index = np.where(self.class_ids == 1)[0][0]
        mask = self.masks[:, :, strip_index]

        for c in range(3):
            img_copy[:, :, c] = np.where(mask != 1,
                                         img_copy[:, :, c] * 0,
                                         img_copy[:, :, c])
        return img_copy

    def cut_strip(self, width=100, height=1000, threshold=0):
        img_copy = self.img.copy()
        black_bg = self.black_bg()

        gray = cv2.cvtColor(black_bg, cv2.COLOR_RGB2GRAY)
        ret, thr = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour_size = cv2.contourArea(contour)
            if contour_size >= 400:
                rect = cv2.minAreaRect(contour)
                box = np.int0(cv2.boxPoints(rect))

        strip_box = np.argsort((self._arrange(box)))

        left_top, right_top, left_bottom, right_bottom = box[strip_box]
        pts1 = np.float32([left_top, left_bottom, right_top, right_bottom])
        pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)

        img_strip = cv2.warpPerspective(img_copy, mtrx, (width, height))
        return img_strip

    def white_balance(self, img):

        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        avg_a = np.average(img_lab[:,:,1])
        avg_b = np.average(img_lab[:,:,2])

        img_lab[:,:,1] = img_lab[:,:,1] - ((avg_a - 128) * (img_lab[:,:,0] / 255.0) * 1.1)
        img_lab[:,:,2] = img_lab[:,:,2] - ((avg_b - 128) * (img_lab[:,:,0] / 255.0) * 1.1)
        img_wb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
        return img_wb