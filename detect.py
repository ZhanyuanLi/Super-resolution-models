# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()  # 获取当前detect.py所在的绝对路径，例如：F:\yolov5\detect1.py
ROOT = FILE.parents[0]  # YOLOv5 root directory。获取detect.py的父目录，例如：F:\yolov5
if str(ROOT) not in sys.path:  # 判断F:\yolov5的路径是否存在这个模块的查询路径中
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative。将上面ROOT的绝对路径转变成一个相对路径

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    # 第一部分：对传入的source进行了一系列判断
    source = str(source)  # source代表data\\images\\bus.jpg这个传入路径，str()：强制转换成字符串路径
    save_img = not nosave and not source.endswith('.txt')  # save inference images。判断是否保存结果
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # 判断传入的路径是否是一个文件地址。suffix：表示后缀，[1:]表示从jpg的j开头，IMG_FORMATS：表示图片的一些格式，VID_FORMATS：表示视频的格式
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))  # 判断给的地址是不是一个网络流地址或网络图片地址。startswith：判断是不是以网络流的地址开头的
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)  # 判断source是不是摄像头、streams或网络流。isnumeric：判断是不是一个数值（有时候传入的可能是"--source 0"，0表示打开电脑上的第一个摄像头）
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:  # 判断传入地址是不是网络流并且是文件
        source = check_file(source)  # download。根据传入的网络流地址去下载图片或视频
    
    # 第二部分：新建了一个保存结果的文件夹
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run。increment_path检测run\detect文件下的exp数字到几，每执行一次都会保存一次增量结果
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir。在run\detect\exp文件下新建一个'labels'文件夹
    
    # 第三部分：负责加载模型的权重
    # Load model
    device = select_device(device)  # 根据代码环境去选择一个加载模型的设备，GPU还是CPU
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # 选择模型的后端框架，是使用pytorch还是其他的框架的权重加载方式去加载模型
    stride, names, pt = model.stride, model.names, model.pt  # 从加载的模型中读取一些值，如：模型的步长，模型检测出来的类别名，模型框架类型是否是pytorch
    imgsz = check_img_size(imgsz, s=stride)  # check image size。检查图像尺寸（一般是640）是否是步长（一般是32）的倍数，如果是的话"imgsz"还是"640×640"；不是的话默认计算一个修改后的图片尺寸
    
    # 第四部分：加载带预测的图片
    # Dataloader
    bs = 1  # batch_size。表示每次输入1张图片
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # 记载图片模块
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    # 第五部分：执行模型的推理过程，图片送入模型，产生预测结果，并将检测框给画出来
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup：初始化一张空白图片，传入模型中，让模型执行了一次前馈传播，相当于让GPU做了一次热身
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())  # seen、dt存储了一些中间结果信息，"dt"存储了每一步的耗时
    for path, im, im0s, vid_cap, s in dataset:  # 遍历dataset，每次执行for循环时，都会执行"LoadImages"中的__next__函数
        # 对图片做了预处理
        # t1 = time_sync()
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)  # torch.Size([3, 640, 480])。"dataset"得到的图片是numpy数组，在模型运算中，必须转成pytorch支持的tensor类型
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0。进行归一化
            if len(im.shape) == 3:  # 判断输入图片尺寸是不是3维
                im = im[None]  # expand for batch dim。缺少batch这个维度，所以扩增1维度，[1, 3, 640, 480]
        # t1 = time_sync()
        # dt[0] += t2 - t1
        
        # 对图片做了预测
        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False  # "visualize"是true的话，会把推断过程中的一些特征图保存下来。
            pred = model(im, augment=augment, visualize=visualize)  # "augment"表示推断的时候是不是要做一个数据增强，"pred"检测出来的框torch.Size([1, 18900, 85])，18900表示检测出来框的数量（后面会进一步过滤），85表示yolov5预训练权重输出的85个预测信息，分别表示4个坐标信息+1个置信度信息+80个类别的概率值

        # NMS。非极大值抑制
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  # 根据"conf_thres"置信度阈值和"iou_thres"去过滤框；"max_det"一张图里最多检测多少个目标；结果：1，5，6；其中5是从18900中降低到了5个目标，6前4个值表示的是坐标点的信息（检测框左上角点的xy，右下角的xy值）、置信度信息、目标所属类别。

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
           
        # 对得到的检测框进行后序处理，把所有的检测框画到了原图中
        # Process predictions
        for i, det in enumerate(pred):  # per image。遍历每个batch中的一个图片，"det"表示5个检测框的6个预测信息([5, 6])
            seen += 1  # 相当于计数，遍历图片的数量
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)  # "frame"会根据"dataset"中有没有"frame"这个属性，没有的话就是0

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg。图片的存储路径，如:runs\\detect\\exp3, "bus.jpg"
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string。拼接到"s"的信息，表图片尺寸
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh。获取原图的高宽
            imc = im0.copy() if save_crop else im0  # for save_crop。检测框的区域是否要裁剪下来单独保存
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # 绘图的工具，在原图上画检测框和标签名
            if len(det):  # 判断是否有检测框
                # Rescale boxes from img_size to im0 size。坐标映射，预测出来的框的位置映射到原图，因为图片经过resize处理
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class。统计所有框的类别
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string。添加到"s"

                # Write results。是否保存结果
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class。获取类别
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 是否画标签。"hide_conf"指画不画置信度
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()  # 返回画好的图片
            if view_img:  # 是否在窗口显示图片
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    
    # 第六部分：打印出一些输出信息
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image。总计每张图片的平均时间
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand。对这个imgsz这个参数做了额外判断，[640]-->[640, 640]
    print_args(vars(opt))  # 打印出了所有参数信息
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))  # 检测"requirements.txt"中的依赖包是否成功安装
    run(**vars(opt))  # 执行后续一系列图片加载、预测、结果保存等流程


if __name__ == '__main__':
    opt = parse_opt()  # 解析参数的函数
    main(opt)


# Terminal窗口下执行命令：F:\yolov5\detect1.py --source data\\images\\bus.jpg
