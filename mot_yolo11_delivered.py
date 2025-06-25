import cv2
import numpy as np
import motmetrics as mm
import csv
from ultralytics import YOLO
import time


# ==================== 0. 辅助函数 ====================
def load_gt_csv(gt_file_path):
    # 从CSV文件加载GT数据
    gt_data = {}
    try:
        with open(gt_file_path, 'r', newline='') as f:
            reader = csv.DictReader(f);
            for row in reader:
                frame_id = int(row['frame']); obj_id = int(row['id']); x = int(row['x']); y = int(row['y']); width = int(row['width']); height = int(row['height']); bbox = [x, y, width, height]
                if frame_id not in gt_data: gt_data[frame_id] = []
                gt_data[frame_id].append((obj_id, bbox))
    except FileNotFoundError: print(f"错误：找不到GT文件 '{gt_file_path}'"); return None
    except KeyError as e: print(f"错误：CSV缺少列: {e}"); return None
    return gt_data

def iou(boxA, boxB):
    # 计算两个边界框的交并比（IoU）
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1]); xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]); yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3]); interArea = max(0, xB - xA) * max(0, yB - yA); boxAArea = boxA[2] * boxA[3]; boxBArea = boxB[2] * boxB[3]
    if boxAArea + boxBArea - interArea == 0: return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

# ==================== 1. 用户配置 ====================
video_path = r"final_01.mp4"
gt_file_path = "gt_final_01.csv"

# 【YOLO配置】 YOLO权重文件的路径
yolo_model_path = r".\yoloFTpt\mixdata_yolo11s\weights\best.pt" 

# YOLO检测的目标类别ID。对于标准的COCO数据集，'person'是类别0。
TARGET_CLASS_ID = 0 
CONFIDENCE_THRESHOLD = 0.5 

window_name = f'YOLOv11 Tracker Evaluation'

# ==================== 2. 初始化 ====================
print("正在初始化...")
# 加载YOLO模型
print(f"正在加载YOLO模型: {yolo_model_path}...")
try:
    model = YOLO(yolo_model_path)
    print("YOLO模型加载成功。")
except Exception as e:
    print(f"错误：加载YOLO模型失败。错误信息: {e}")
    exit()

cap = cv2.VideoCapture(video_path)
gt_data = load_gt_csv(gt_file_path)
if gt_data is None: exit()

acc = mm.MOTAccumulator(auto_id=True)
is_initialized = False
last_pred_box = None

# 用于FPS计算的变量
frame_count = 0
total_time = 0

# ==================== 3. 跟踪与评测主循环 ====================
print("初始化完成，开始使用YOLO进行跟踪和评测...")
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_idx += 1
    start_time = time.time() # 记录开始时间

    # --- 使用YOLO进行检测 ---
    # results[0].boxes.xywh 返回的是 [cx, cy, w, h] 格式, 我们需要的是 [x, y, w, h] (左上角坐标)
    results = model(frame, verbose=False) # verbose=False禁止打印详细信息
    detections = []
    for r in results:
        for box in r.boxes:
            if int(box.cls) == TARGET_CLASS_ID and float(box.conf) > CONFIDENCE_THRESHOLD:
                x_center, y_center, w, h = box.xywh[0].cpu().numpy().tolist()
                x_left = x_center - w / 2
                y_top = y_center - h / 2
                detections.append([int(x_left), int(y_top), int(w), int(h)])

    # --- 跟踪逻辑 (Tracking-by-Detection) ---
    final_pred_box = None
    if not is_initialized and frame_idx in gt_data:
        # 第一次初始化：
        initial_gt_box = gt_data[frame_idx][0][1]
        best_iou = -1
        for det_box in detections:
            current_iou = iou(initial_gt_box, det_box)
            if current_iou > best_iou:
                best_iou = current_iou
                final_pred_box = det_box
        
        if final_pred_box:
            is_initialized = True
            last_pred_box = final_pred_box
            print(f"YOLO跟踪器在第 {frame_idx} 帧初始化成功。")

    elif is_initialized:
        # 后续帧：找到与上一帧预测框IoU最高的检测结果
        best_iou = -1
        best_det_box = None
        if last_pred_box and detections:
            for det_box in detections:
                current_iou = iou(last_pred_box, det_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_det_box = det_box
        
        # 只有在找到匹配时才更新，否则认为跟丢
        if best_det_box is not None:
            final_pred_box = best_det_box
            last_pred_box = final_pred_box

    # --- FPS计算 ---
    end_time = time.time()
    total_time += (end_time - start_time)
    frame_count += 1
    
    # --- MOTMetrics 更新 ---
    if frame_idx in gt_data:
        gt_items = gt_data[frame_idx]
        gt_ids = [item[0] for item in gt_items]
        gt_bboxes = [item[1] for item in gt_items]
        
        pred_ids = []
        pred_bboxes = []
        if final_pred_box is not None:
            pred_ids.append(1) 
            pred_bboxes.append(final_pred_box)

        distances = mm.distances.iou_matrix(gt_bboxes, pred_bboxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances)

    # --- 可视化 ---
    vis = frame.copy()
    if final_pred_box is not None:
        (x, y, w, h) = [int(v) for v in final_pred_box]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(vis, 'YOLO Prediction', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    if frame_idx in gt_data:
        for _, gt_bbox in gt_data[frame_idx]:
            (x_gt, y_gt, w_gt, h_gt) = gt_bbox
            cv2.rectangle(vis, (x_gt, y_gt), (x_gt + w_gt, y_gt + h_gt), (0, 255, 0), 2)
            cv2.putText(vis, 'GroundTruth', (x_gt, y_gt - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    current_fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
    cv2.putText(vis, f"Frame: {frame_idx} | FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow(window_name, vis)

    if cv2.waitKey(1) & 0xFF == 27: break

# ==================== 4. 计算并显示结果 ====================
print("\n视频播放完毕，正在计算评测指标...")
cap.release()
cv2.destroyAllWindows()

# 计算平均FPS
average_fps = frame_count / total_time if total_time > 0 else 0

mh = mm.metrics.create()
summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='overall')
strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)

print(f"\n===== Evaluation Report (model: {yolo_model_path}) =====")
print(strsummary)
print(f"Average FPS: {average_fps:.2f}")