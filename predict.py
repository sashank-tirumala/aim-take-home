import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

class BaseEngine(object):
    """
    Base class for TensorRT engine
    """
    def __init__(self, engine_path, imgsz=(640,640)):
        self.imgsz = imgsz
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger,'')
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
                
    def infer(self, img):
        """
        Run inference on the TensorRT engine
        Inputs:
            img: numpy array of shape (1, h, w, 3)
        Outputs:
            data: list of numpy arrays indicating the YOLO model predictions
        """
        self.inputs[0]['host'] = np.ravel(img)
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        data = [out['host'] for out in self.outputs]
        return data

    def inference(self, img):
        """
        Run inference on the TensorRT engine
        Inputs:
            img: numpy array of shape (1, h, w, 3)
        Outputs:
            final_boxes: numpy array of shape (num, 4)
        """
        origin_img = img
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        img, ratio = preproc(origin_img, self.imgsz)
        num, final_boxes, final_scores, final_cls_inds = self.infer(img)
        final_boxes = np.reshape(final_boxes, (-1, 4))
        num = num[0]
        if num >0:
            final_boxes, final_scores, final_cls_inds = final_boxes[:num]/ratio, final_scores[:num], final_cls_inds[:num]
            idxs = final_cls_inds == 32
            final_boxes = final_boxes[idxs]
        return final_boxes

def preproc(image, input_size, swap=(2, 0, 1)):
    """
    Preprocess an image before TRT YOLO inferencing
    Inputs:
        image: numpy array of shape (h, w, 3)
        input_size: tuple of (h, w)
        swap: tuple of (r, g, b)
    Outputs:
        padded_img: numpy array of shape (input_size[0], input_size[1], 3)
        r: ratio of original image to padded image
    """
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def vis(img, box):
    """
    Visualize the bounding box on the image
    Inputs:
        img: numpy array of shape (h, w, 3)
        box: numpy array of shape (4,)
    Outputs:
        img: numpy array of shape (h, w, 3)
    """
    if len(box) == 0:
        return img
    x0 = int(box[0])
    y0 = int(box[1])
    x1 = int(box[2])
    y1 = int(box[3])

    color = (np.array([0,0,1]) * 255).astype(np.uint8).tolist()
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    return img

class filter:
    """
    Filter the bounding boxes
    """
    def __init__(self):
        self.center_x=[]
        self.center_y=[]
        self.obs_x1=[]
        self.obs_y1=[]
        self.obs_x2=[]
        self.obs_y2=[]
        self.areas = []
        self.ratios = []
        self.count = 0
    
    def update(self, box):
        """
        Update the filter, based on area, ratio, and change of center
        Inputs:
            box: numpy array of shape (4,)
        Outputs:
            box: numpy array of shape (4,)
        """
        centerx = (box[0]+box[2])/2
        if self.count:
            if np.abs(centerx - self.center_x[-1]) > 260:
                return []
        centery = (box[1]+box[3])/2
        area = (box[2]-box[0])*(box[3]-box[1])
        if area < 1000:
            return []
        ratio = (box[3]-box[1])/(box[2]-box[0])
        if ratio > 1.8:
            return []
        self.ratios.append(ratio)
        self.areas.append(area)
        self.obs_x1.append(box[0])
        self.obs_y1.append(box[1])
        self.obs_x2.append(box[2])
        self.obs_y2.append(box[3])
        self.center_x.append(centerx)
        self.center_y.append(centery)
        self.count=1
        return box
    
    def plot(self):
        """
        Plot the center, area, and ratio
        """
        plt.plot(self.center_x)
        plt.plot(self.areas)
        plt.plot(self.ratios)
        plt.show()



if __name__ == "__main__":
    pred = BaseEngine(engine_path='yolov7-nms-32.trt')
    video_path = '/home/sashank/projects/personal/aim_take_home/input/ball_tracking_video.mp4'
    cap = cv2.VideoCapture(video_path)
    output_path = 'output_video.mp4'
    # Get the video dimensions and frame rate
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Create a VideoWriter object to write the frames to a new video file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


    if not cap.isOpened():
        print("Error: Could not open the video file.")
        exit()
    frames=[]
    start_time = time.time()
    boxes = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        box_curr = pred.inference(frame)
        boxes.append(box_curr)
        frames.append(frame)
    end_time = time.time()
    print(end_time-start_time)
    x_coords1 = []
    y_coords1 = []
    x_coords2 = []
    y_coords2 = []
    filter1 = filter()
    for i in range(len(frames)):
        frame = frames[i]
        box = boxes[i]
        if len(box) != 0:
            box = filter1.update(box[0])
            frame = vis(frame, box)
        out.write(frame)
    filter1.plot()
    cap.release()
    out.release()
    pass