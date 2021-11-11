import cv2

import numpy as np


WIN_NAME = 'Crop Selector'


class CropSelector(object):

    def __init__(self, in_path, app_scale=0.4):
        cv2.namedWindow(WIN_NAME)
        cv2.setMouseCallback(WIN_NAME, self.click_handler)
        self.img = cv2.imread(in_path)
        self.img = cv2.resize(self.img, (0, 0), fx=app_scale, fy=app_scale)
        self.pt1 = None
        self.pt2 = None

        self.state = 'SLEEP'
        self.app_scale = app_scale


    @staticmethod
    def int_scale(x, scale):
        return np.round((np.array(x) / scale), 0).astype(int)


    def click_handler(self, event, x, y, *_):

        if event == cv2.EVENT_LBUTTONDOWN and self.state == 'SLEEP':
            self.pt1 = (x, y)
            self.state = 'START'

        elif event == cv2.EVENT_LBUTTONDOWN and self.state != 'SLEEP':
            self.pt1 = (x, y)
            self.pt2 = None
            self.state = 'START'

        elif event == cv2.EVENT_MOUSEMOVE and self.state == 'START':
            self.pt2 = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.state == 'START':
            self.pt2 = (x, y)
            self.state = 'DONE'
            w = self.pt2[0] - self.pt1[0]
            h = self.pt2[1] - self.pt1[1]
            side = max(h, w)

            pt1 = self.int_scale(self.pt1, scale=self.app_scale)
            pt2 = self.int_scale(self.pt2, scale=self.app_scale)
            print(f'crop_bbox = [{pt1[0]}, {pt1[1]}, {pt2[0]}, {pt2[1]}]')

            if h < w:
                delta = abs(h - side)
                if delta % 2 == 1:
                    delta += 1
                self.pt1 = self.pt1[0], self.pt1[1] - delta // 2
                self.pt2 = self.pt1[0] + side, self.pt1[1] + side
            elif h > w:
                delta = abs(w - side)
                if delta % 2 == 1:
                    delta += 1
                self.pt1 = self.pt1[0] - delta // 2, self.pt1[1]
                self.pt2 = self.pt1[0] + side, self.pt1[1] + side

            pt1 = self.int_scale(self.pt1, scale=self.app_scale)
            pt2 = self.int_scale(self.pt2, scale=self.app_scale)
            side = pt2[1] - pt1[1]
            print(f'(square version): '
                  f'crop_x_min={pt1[0]}, '
                  f'crop_y_min={pt1[1]}, '
                  f'crop_side={side}')


    def draw_selection(self):
        if self.pt1 is not None and self.pt2 is not None:
            return cv2.rectangle(self.img.copy(), self.pt1, self.pt2, color=(64, 0, 255), thickness=8)
        return self.img


    def run(self):
        while True:

            img2show = self.draw_selection()
            cv2.imshow(WIN_NAME, img2show)
            key = cv2.waitKey(30)
            if key == 27:
                self.state = 'SLEEP'
                self.pt1 = None
                self.pt2 = None
            elif key == 113:
                exit(0)
            elif key != -1:
                print(f'$> pressed KEY={key}')


def main():
    CropSelector(in_path='debug/debug_img_00.png').run()


if __name__ == '__main__':
    main()
