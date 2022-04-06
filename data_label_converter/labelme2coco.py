# labelme格式标注信息转换为coco格式标注信息
import json
import os
import random
import time

import cv2
import numpy as np


class Labelme2COCO(object):
    def __init__(self, folders,
                 class_names,
                 coco_dir,
                 coco_year="2017",
                 train_ratio=0.9090,
                 get_rotate_path=True,
                 do_resize=True,
                 target_shape=(512, 512)):
        """
        :param folders: labelme格式的文件夹路径(list[str])
        :param class_names: 类别名称(不需要包含"background"类)
        :param coco_dir: coco格式的文件夹
        :param coco_year: coco数据集的年份
        :param train_ratio: 训练集和验证集的比例
        :param get_rotate_path: 是否获取旋转后的图片路径
        :param do_resize: 是否进行图片的resize
        :param target_shape: 如果图片进行resize, 则指定目标图片的大小(h, w)
        """
        self.folders = folders
        self.class_names = class_names
        self.coco_dir = coco_dir
        self.coco_year = coco_year
        self.train_ratio = train_ratio
        self.get_rotate_path = get_rotate_path
        self.do_resize = do_resize
        self.target_shape = target_shape

        self.coco_img_dir_train = os.path.join(
            self.coco_dir, "train{}".format(self.coco_year))
        self.coco_img_dir_val = os.path.join(
            self.coco_dir, "val{}".format(self.coco_year))
        self.coco_ann_dir = os.path.join(self.coco_dir, "annotations")
        self.coco_ann_file_train = os.path.join(
            self.coco_ann_dir, "instances_{}_train.json".format(self.coco_year))
        self.coco_ann_file_val = os.path.join(
            self.coco_ann_dir, "instances_{}_val.json".format(self.coco_year))

        self.dict_name_id = {}
        for i, class_name in enumerate(self.class_names):
            self.dict_name_id[class_name] = i + 1

        self.coco_categories = []
        for k, v in self.dict_name_id.items():
            self.coco_categories.append({"supercategory": "none",
                                         "id": v,
                                         "name": k})

        self.coco_info = {"description": "none",
                          "url": "none",
                          "version": "none",
                          "year": self.coco_year,
                          "contributor": "none",
                          "date_created": time.strftime("%Y-%m-%d %H:%M:%S",
                                                        time.localtime())}

        self.coco_licenses = [{"url": "none",
                               "id": 1,
                               "name": "none"}]

        self.coco_image_id = 0
        self.coco_annotation_id = 0
        self.coco_images = []
        self.coco_annotations = []

    def mkdir_coco(self):
        """
        创建coco格式的文件夹
        """
        if not os.path.exists(self.coco_dir):
            os.mkdir(self.coco_dir)
        if not os.path.exists(self.coco_ann_dir):
            os.mkdir(self.coco_ann_dir)
        if not os.path.exists(self.coco_img_dir_train):
            os.mkdir(self.coco_img_dir_train)
        if not os.path.exists(self.coco_img_dir_val):
            os.mkdir(self.coco_img_dir_val)

    def split_trainval(self):
        """
        将图片分为训练集和验证集
        """
        train_img_paths = []
        val_img_paths = []
        # random.seed(4)
        for folder in self.folders:
            img_paths = self.get_img_paths(folder)
            random.shuffle(img_paths)
            train_num = int(len(img_paths) * self.train_ratio)
            train_img_paths.extend(img_paths[:train_num])
            val_img_paths.extend(img_paths[train_num:])

        if self.get_rotate_path:
            for img_path in train_img_paths:
                img_dir, img_name = os.path.split(img_path)
                img_path_r90, img_path_r180, img_path_r270 = self.get_rotate_img_paths(
                    img_dir, img_path)
                if os.path.exists(img_path_r90):
                    train_img_paths.append(img_path_r90)
                if os.path.exists(img_path_r180):
                    train_img_paths.append(img_path_r180)
                if os.path.exists(img_path_r270):
                    train_img_paths.append(img_path_r270)
            for img_path in val_img_paths:
                img_dir, img_name = os.path.split(img_path)
                img_path_r90, img_path_r180, img_path_r270 = self.get_rotate_img_paths(
                    img_dir, img_path)
                if os.path.exists(img_path_r90):
                    val_img_paths.append(img_path_r90)
                if os.path.exists(img_path_r180):
                    val_img_paths.append(img_path_r180)
                if os.path.exists(img_path_r270):
                    val_img_paths.append(img_path_r270)
        random.shuffle(train_img_paths)
        random.shuffle(val_img_paths)

        return train_img_paths, val_img_paths

    def parse_labelme_json(self, img_path, save_dir):
        """
        解析labelme格式的json文件
        :param img_path: 图片路径
        :param save_dir: 图片保存的目录
        :return:
        """
        print("||img_id:{}||ann_id:{}||{}".format(
            self.coco_image_id, self.coco_annotation_id, img_path))

        img_dir, img_name = os.path.split(img_path)
        name_font, name_end = os.path.splitext(img_name)
        save_img_name = "{}_{}{}".format(
            name_font, self.coco_image_id, name_end)
        save_img_path = os.path.join(save_dir, save_img_name)

        # ----------处理图像文件----------
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if self.do_resize:
            scale, (pad_left, pad_top), img_resize_pad = self.resize_padding(img,
                                                                             self.target_shape[0],
                                                                             self.target_shape[1],
                                                                             interpolation=cv2.INTER_LINEAR,
                                                                             pad_value=255)
        else:
            scale = 1.0
            pad_left = pad_top = 0
            img_resize_pad = img
        self.coco_images.append({
            "license": 1,
            "coco_url": "none",
            "date_captured": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "height": img_resize_pad.shape[0],
            "width": img_resize_pad.shape[1],
            "id": self.coco_image_id})
        cv2.imencode('.jpg', img_resize_pad)[1].tofile(save_img_path)
        self.coco_image_id += 1

        # ----------处理标注信息文件----------
        json_path = os.path.splitext(img_path)[0]+'.json'
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            for shape in json_data['shapes']:
                if shape['label'] not in self.class_names:
                    continue
                if shape['shape_type'] == 'rectangle':
                    xmin = min([xy[0] for xy in shape["points"]])
                    xmax = max([xy[0] for xy in shape["points"]])
                    ymin = min([xy[1] for xy in shape["points"]])
                    ymax = max([xy[1] for xy in shape["points"]])
                    points = np.array([[xmin, ymin], [xmax, ymin],
                                       [xmax, ymax], [xmin, ymax]],
                                      dtype=np.float32)
                if shape['shape_type'] == 'polygon':
                    points = np.array(shape['points'], dtype=np.float32)

                points[:, 0] = points[:, 0] * scale + pad_left
                points[:, 1] = points[:, 1] * scale + pad_top
                xmin = points[:, 0].min()
                xmax = points[:, 0].max()
                ymin = points[:, 1].min()
                ymax = points[:, 1].max()
                segm = points.reshape(1, -1)
                if (xmax == xmin) or (ymax == ymin):
                    continue
                self.coco_annotations.append({
                    "iscrowd": 0,
                    "image_id": self.coco_image_id,
                    "id": self.coco_annotation_id,
                    "category_id": self.dict_name_id[shape['label']],
                    "area": (xmax - xmin) * (ymax - ymin),
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "segmentation": segm.tolist()})
                self.coco_annotation_id += 1

    def to_coco(self, img_paths, is_training):
        coco_dict_data = {}
        coco_dict_data['info'] = self.coco_info
        coco_dict_data['licenses'] = self.coco_licenses
        coco_dict_data['type'] = 'instances'
        coco_dict_data['categories'] = self.coco_categories

        if is_training:
            save_dir = self.coco_img_dir_train
            save_json_path = self.coco_ann_file_train
        else:
            save_dir = self.coco_img_dir_val
            save_json_path = self.coco_ann_file_val

        self.coco_images.clear()
        self.coco_annotations.clear()
        for img_path in img_paths:
            self.parse_labelme_json(img_path, save_dir)

        coco_dict_data['images'] = self.coco_images
        coco_dict_data['annotations'] = self.coco_annotations
        with open(save_json_path, 'w') as f:
            json.dump(coco_dict_data, f)
        print("====", save_json_path, "json saved.")

    def run(self):
        self.mkdir_coco()
        train_img_paths, val_img_paths = self.split_trainval()
        self.to_coco(train_img_paths, is_training=True)
        self.to_coco(val_img_paths, is_training=False)
        print("=" * 30)
        print("总数据量：", len(train_img_paths) + len(val_img_paths))
        print("训练集数据量：", len(train_img_paths))
        print("验证集数据量：", len(val_img_paths))
        print("=" * 30)

    @ staticmethod
    def resize_padding(img_array, target_h, target_w,
                       interpolation=cv2.INTER_LINEAR,
                       pad_value=255):
        """
        对图片进行resize和padding
        :param img_array: 图片路径
        :param target_h: 目标高度
        :param target_w: 目标宽度
        :param interpolation: 填充方式
        :param pad_value: 填充的值
        :return:
            scale: 缩放比例
            (pad_left, pad_top): 左侧和顶部的填充像素数
            img_resize_pad: 缩放后的图片
        """
        ori_h, ori_w, _ = img_array.shape
        scale = min(target_h / ori_h, target_w / ori_w)
        new_h, new_w = round(ori_h * scale), round(ori_w * scale)
        img_resize = cv2.resize(img_array, (new_w, new_h),
                                interpolation=interpolation)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        img_resize_pad = cv2.copyMakeBorder(img_resize,
                                            pad_top, target_h - new_h - pad_top,
                                            pad_left, target_w - new_w - pad_left,
                                            cv2.BORDER_CONSTANT,
                                            value=(pad_value, pad_value, pad_value))
        return scale, (pad_left, pad_top), img_resize_pad

    @ staticmethod
    def get_img_paths(folder):
        """获取folder下的各级目录中图像的路径"""
        img_paths = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    img_paths.append(os.path.join(root, file))
        return img_paths

    @ staticmethod
    def get_rotate_img_paths(folder, img_path):
        folder_r = folder+"_rotate"
        dir_o, img_name = os.path.split(img_path)
        dir_r = dir_o.replace(folder, folder_r)

        name_font, name_end = os.path.splitext(img_name)
        img_name_r90 = "{}_r90{}".format(name_font, name_end)
        img_name_r180 = "{}_r180{}".format(name_font, name_end)
        img_name_r270 = "{}_r270{}".format(name_font, name_end)

        img_path_r90 = os.path.join(dir_r, img_name_r90)
        img_path_r180 = os.path.join(dir_r, img_name_r180)
        img_path_r270 = os.path.join(dir_r, img_name_r270)

        return img_path_r90, img_path_r180, img_path_r270


if __name__ == "__main__":
    """
        folders: labelme格式的文件夹路径(list[str])
        class_names: 类别名称(不需要包含"background"类)
        coco_dir: coco格式的文件夹路径([对应coco_year]会在目录下新建"train2017", "val2017", "annotations",文件夹
                在"annotations"文件夹下新建"instances_train2017.json", "instances_val2017.json"文件)
        coco_year: coco数据集的年份(默认2017)
        train_ratio: 训练集和验证集的比例(默认0.9090)
        get_rotate_path: 是否获取旋转后的图片路径, 如下例:
                (/a/b/c.jpg -> /a/b_rotate/c_r90.jpg, /a/b_rotate/c_r180.jpg, /a/b_rotate/c_r270.jpg)
        do_resize: 是否进行图片的resize
        target_shape: 如果图片进行resize, 则指定目标图片的大小(h, w)
    """
    # -----------------------------印章、条码、二维码、手写签名-----------------------------
    folders = ["/a/b/c/labelme_data",
                        "/a/b/c/labelme_data_new"]
    coco_dir = "a/b/c/coco"
    coco_year = "2017"
    class_names = ['Seal', 'QRcode', 'Barcode', 'HWSign']
    train_ratio = 0.9090
    get_rotate_path = True
    do_resize = True
    target_shape = (1024, 1024)
    # ---------------------------------------------------------------------------------
    train_scale = 0.9090
    Converter = Labelme2COCO(folders=folders,
                             class_names=class_names,
                             coco_dir=coco_dir,
                             coco_year=coco_year,
                             train_ratio=train_ratio,
                             get_rotate_path=get_rotate_path,
                             do_resize=do_resize,
                             target_shape=target_shape)
    Converter.run()
