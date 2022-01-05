import numpy as np
import cv2

class_mapping = {
    0: 'No mask ',
    1: 'Mask '
}

def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    conf = boxes[:,4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(conf)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
        np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("float")




def resize(img, desired_size=640):
    old_size = img.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    img_new = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    img_new = cv2.copyMakeBorder(img_new, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return img_new


def model_predict(model, img_arr):
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    predictions = [0] * len(img_arr)
    for i, val in enumerate(predictions):
        model.set_tensor(input_details[0]['index'], img_arr[i].reshape((1, 640, 640, 3)))
        model.invoke()
        predictions[i] = model.get_tensor(output_details[0]['index'])

    predictions_bb = np.array(predictions)
    
    return predictions_bb


def draw_bb(img, res, name):
    dw, dh = img.shape[:2]
    img_draw = img.copy()
    for bb in res[:10]:
        x, y, w, h, conf, cls0, cls1 = bb
        cls = 0 if cls0 > cls1 else 1
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)

        l = max(0, l)
        r = min(r, dw-1)
        t = max(t, 0)
        b = min(b, dh-1)
        img_draw = cv2.rectangle(img_draw, (l, t), (r, b), (255, 0, 0))
        cv2.putText(img_draw, class_mapping[cls] + str(conf), (l, t))

    cv2.imwrite('/static/' + name, img_draw)