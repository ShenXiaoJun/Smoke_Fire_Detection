import sys
import argparse

import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.general import *
from utils import torch_utils


class Smoke_File_Detector():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='weights/smoke.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        self.opt = parser.parse_args()

        self.device = torch_utils.select_device(self.opt.device)

        # Initialize
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA   

        # Load models
        self.model = attempt_load(self.opt.weights, map_location=self.device)
        self.imgsz = check_img_size(self.opt.img_size, s=self.model.stride.max())
        if self.half:
            self.model.half()  

    # 本地调用
    def detect_test(self,test_list):
        for i,img in enumerate(test_list):
            im0 = img
            img = letterbox(img, new_shape=self.opt.img_size)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)  # faster

            # Run inference
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            if i == 0:
                batch_img = img
            else:
                batch_img = torch.cat([batch_img,img],axis = 0)

        pred = self.model(batch_img, augment=self.opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)

        # Process detections
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        batch_results = []
        for i, det in enumerate(pred):  # detections per image
            results = []
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(batch_img.shape[2:], det[:, :4], im0.shape).round()                
                #det = det.cuda().data.cpu().numpy()
                det = det.cpu().numpy() #shenxj use cpu
                for *xyxy, conf, cls in det:
                    w = xyxy[2]-xyxy[0]
                    h = xyxy[3]-xyxy[1]
                    result = {'bbox':xyxy, 'label':names[int(cls)], 'conf':conf}
                    results.append(result)

            batch_results.append(results)

        # print(batch_results)
        return batch_results 

    # server调用               
    def detect(self, **kwargs):
        params = kwargs
        test_list = [params["img"]]
        for i,img in enumerate(test_list):
            im0 = img
            img = letterbox(img, new_shape=self.opt.img_size)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            # Run inference
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            if i == 0:
                batch_img = img
            else:
                batch_img = torch.cat([batch_img,img],axis = 0)

        pred = self.model(batch_img, augment=self.opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)

        # Process detections
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        batch_results = []
        for i, det in enumerate(pred):  # detections per image
            results = []
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(batch_img.shape[2:], det[:, :4], im0.shape).round()                
                det = det.cuda().data.cpu().numpy()
                for *xyxy, conf, cls in det:
                    w = xyxy[2]-xyxy[0]
                    h = xyxy[3]-xyxy[1]
                    result = {'bbox':xyxy, 'label':names[int(cls)], 'conf':conf}
                    results.append(result)

            batch_results.append(results)

        # print(batch_results)
        return batch_results         


if __name__ == "__main__":
    import cv2
    img = cv2.imread('../doc/demo.jpg')

    det = Smoke_File_Detector()
    print(det.detect_test([img]))
    for detection in det.detect_test([img])[0]:
        #x1, y1, x2, y2, conf, class_id = detection[:6]  # 获取边界框、置信度、类别ID
        #label = model.names[int(class_id)]  # 获取类别名称
        #confidence = conf.item()  # 将 tensor 转换为普通数值

        #shenxj+++
        bbox = detection['bbox']
        label = detection['label']
        conf = detection['conf']
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        #shenxj---

        # 绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # 在图像上写类别标签和置信度
        text = f'{label} {conf:.2f}'
        cv2.putText(img, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 6. 保存合并后的图像
    output_path = '/var/www/html/yolov5/output_image.jpg'  # 保存路径
    cv2.imwrite(output_path, img)
