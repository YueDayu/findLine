import numpy as np
import cv2
import os


image_path = 'F:/dataset/origin'
all_file = [os.path.join(image_path, x) for x in os.listdir(image_path)]

image_save_path = 'F:/dataset/image'
label_save_path = 'F:/dataset/label'

current_file = 0
current_id = 0
all_pt_lists = []
pt_list = []
dst_list = np.array([(0, 0), (0, 54), (104, 54), (104, 0)], dtype='float32')
current_img = np.zeros((105, 55))
line_img = np.zeros((105, 55))
save_img = np.zeros((105, 55))
all_img = cv2.imread(all_file[0], flags=cv2.IMREAD_GRAYSCALE)
show_img = np.zeros_like(all_img)
cv2.namedWindow('image')
cv2.namedWindow('detail')

is_l_down = False


def draw_img():
    global show_img
    global pt_list
    global all_pt_lists
    global all_img
    show_img = np.zeros((all_img.shape[0], all_img.shape[1], 3))
    show_img[:, :, 0] = all_img / 255
    show_img[:, :, 1] = all_img / 255
    show_img[:, :, 2] = all_img / 255
    show_img = cv2.resize(show_img, (int(all_img.shape[0] * 0.68), int(all_img.shape[1] * 0.68)))
    for x in all_pt_lists:
        for i in range(len(x), 0, -1):
            cv2.line(show_img, x[i - 1], x[i - 2], (0, 100, 0), 2)
    for x in pt_list:
        cv2.circle(show_img, x, 2, (0, 0, 100), -1)
    for i in range(len(pt_list) - 1):
        cv2.line(show_img, pt_list[i], pt_list[i + 1], (0, 0, 100), 2)
    if len(pt_list) == 4:
        cv2.line(show_img, pt_list[0], pt_list[-1], (0, 0, 100), 2)


def choose_mouse_handler(event, x, y, flags, param):
    global pt_list
    if event == cv2.EVENT_LBUTTONUP:
        if len(pt_list) < 4:
            pt_list.append((x, y))
    if event == cv2.EVENT_RBUTTONUP:
        if len(pt_list) > 0:
            pt_list = pt_list[:-1]
    draw_img()


def draw_mouse_handler(event, x, y, flags, param):
    global current_img
    global line_img
    global is_l_down
    if event == cv2.EVENT_LBUTTONDOWN:
        is_l_down = True
    if event == cv2.EVENT_LBUTTONUP:
        is_l_down = False
    if event == cv2.EVENT_MOUSEMOVE and is_l_down:
        cv2.circle(current_img, (x, y), 2, (0, 0, 100), -1)
        line_img[int(y / 2), int(x / 2)] = 255


def main_key_handler(key):
    global pt_list
    global all_pt_lists
    global current_img
    global dst_list
    global all_img
    global save_img
    global line_img
    global current_file
    global all_file
    global current_id
    if key == 32:
        if len(pt_list) == 4:
            result = True
            new_list = [(x[0] / 0.68, x[1] / 0.68) for x in pt_list]
            matirx = cv2.getPerspectiveTransform(np.array(new_list, dtype="float32"), dst_list)
            save_img = cv2.warpPerspective(all_img, matirx, (105, 55))
            line_img = np.zeros((55, 105))
            current_img = np.zeros((save_img.shape[0], save_img.shape[1], 3))
            current_img[:, :, 0] = save_img / 255
            current_img[:, :, 1] = save_img / 255
            current_img[:, :, 2] = save_img / 255
            current_img = cv2.resize(current_img, (210, 110))
            while True:
                cv2.imshow('detail', current_img)
                key = cv2.waitKey(20)
                if key == 113:
                    result = False
                    break
                if key == 32:
                    result = True
                    # cv2.imwrite('test.bmp', save_img)
                    cv2.imwrite(os.path.join(image_save_path, str(current_id) + '.bmp'), save_img)
                    # cv2.imwrite('test_line.bmp', line_img)
                    cv2.imwrite(os.path.join(label_save_path, str(current_id) + '.bmp'), line_img)
                    current_id += 1
                    break
                # sub_key_handler(key)
            if result:
                all_pt_lists.append(pt_list)
                pt_list = []
    if key == 2555904:
        if current_file < len(all_file) - 1:
            current_file += 1
            all_img = cv2.imread(all_file[current_file], flags=cv2.IMREAD_GRAYSCALE)
            pt_list = []
            all_pt_lists = []
    if key == 2424832:
        if current_file > 0:
            current_file -= 1
            all_img = cv2.imread(all_file[current_file], flags=cv2.IMREAD_GRAYSCALE)
            pt_list = []
            all_pt_lists = []


cv2.setMouseCallback('image', choose_mouse_handler)
cv2.setMouseCallback('detail', draw_mouse_handler)

draw_img()
while True:
    cv2.imshow('image', show_img)
    key = cv2.waitKey(20)
    if key == 113:
        break
    main_key_handler(key)

cv2.destroyAllWindows()
