import torch


### TO XYXY ###
def box_yxyx_to_xyxy(x):
    """ Converts bounding box with format [y0, x0, y1, x1] to format [x0, y0, x1, y1] """
    y0, x0, y1, x1 = torch.split(x, 1, dim=-1)
    b = [
        x0,
        y0,
        x1,
        y1
    ]
    return torch.cat(b, dim=-1)


def box_xywh_to_xyxy(x):
    """ Converts bounding box with format [x0, y0, w, h] to format [x0, y0, x1, y1] """
    x0, y0, w, h = torch.split(x, 1, dim=-1)
    b = [
        x0,
        y0,
        x0 + w,
        y0 + h
    ]
    return torch.cat(b, dim=-1)


def box_cxcywh_to_xyxy(x):
    """ Converts bounding box with format [center_x, center_y, w, h] to format [x0, y0, x1, y1] """
    cx, cy, w, h = torch.split(x, 1, dim=-1)
    b = [
        cx - 0.5 * w,
        cy - 0.5 * h,
        cx + 0.5 * w,
        cy + 0.5 * h
    ]
    return torch.cat(b, dim=-1)


### TO YXYX ###
def box_xyxy_to_yxyx(x):
    """ Converts bounding box with format [x0, y0, x1, y1] to format [y0, x0, y1, x1] """
    x0, y0, x1, y1 = torch.split(x, 1, dim=-1)
    b = [
        y0,
        x0,
        y1,
        x1
    ]
    return torch.cat(b, dim=-1)


def box_xywh_to_yxyx(x):
    """ Converts bounding box with format [x0, y0, w, h] to format [y0, x0, y1, x1] """
    x0, y0, w, h = torch.split(x, 1, dim=-1)
    b = [
        y0,
        x0,
        y0 + h,
        x0 + w
    ]
    return torch.cat(b, dim=-1)


def box_cxcywh_to_yxyx(x):
    """ Converts bounding box with format [center_x, center_y, w, h] to format [y0, x0, y1, x1] """
    cx, cy, w, h = torch.split(x, 1, dim=-1)
    b = [
        cy - 0.5 * h,
        cx - 0.5 * w,
        cy + 0.5 * h,
        cx + 0.5 * w
    ]
    return torch.cat(b, dim=-1)


### TO XYWH ###
def box_xyxy_to_xywh(x):
    """ Converts bounding box with format [x0, y0, x1, y1] to format [x0, y0, w, h] """
    x0, y0, x1, y1 = torch.split(x, 1, dim=-1)
    b = [
        x0,
        y0,
        x1 - x0,
        y1 - y0
    ]
    return torch.cat(b, dim=-1)


def box_yxyx_to_xywh(x):
    """ Converts bounding box with format [y0, x0, y1, x1] to format [x0, y0, w, h] """
    y0, x0, y1, x1 = torch.split(x, 1, dim=-1)
    b = [
        x0,
        y0,
        x1 - x0,
        y1 - y0
    ]
    return torch.cat(b, dim=-1)


def box_cxcywh_to_xywh(x):
    """ Converts bounding box with format [center_x, center_y, w, h] to format [x0, y0, w, h] """
    cx, cy, w, h = torch.split(x, 1, dim=-1)
    b = [
        cx - 0.5 * w,
        cy - 0.5 * h,
        w,
        h
    ]
    return torch.cat(b, dim=-1)


### TO CXCYWH ###
def box_xyxy_to_cxcywh(x):
    """ Converts bounding box with format [x0, y0, x1, y1] to format [center_x, center_y, w, h] """
    x0, y0, x1, y1 = torch.split(x, 1, dim=-1)
    b = [
        (x0 + x1) / 2,
        (y0 + y1) / 2,
        x1 - x0,
        y1 - y0
    ]
    return torch.cat(b, dim=-1)


def box_yxyx_to_cxcywh(x):
    """ Converts bounding box with format [y0, x0, y1, x1] to format [center_x, center_y, w, h] """
    y0, x0, y1, x1 = torch.split(x, 1, dim=-1)
    b = [
        (x0 + x1) / 2,
        (y0 + y1) / 2,
        x1 - x0,
        y1 - y0
    ]
    return torch.cat(b, dim=-1)


def box_xywh_to_cxcywh(x):
    """ Converts bounding box with format [x0, y0, w, h] to format [center_x, center_y, w, h] """
    x0, y0, w, h = torch.split(x, 1, dim=-1)
    b = [
        x0 + 0.5 * w,
        y0 + 0.5 * h,
        w,
        h
    ]
    return torch.cat(b, dim=-1)