def YoloToReg(img, x, y, w, h):
    '''Convert coordinate from YOLO to 2 points (left, top) (bottom, right)'''
    dh, dw = img.shape[:2]

    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)
    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1
    return l, t, r, b


def RegToYolo(img, l, t, r, b):
    dh, dw = img.shape[0], img.shape[1]
    x = (l+r)/2.0 / dw
    y = (t+b)/2.0 / dh
    w = (r-l) / dw
    h = (b-t) / dh
    return x, y, w, h
