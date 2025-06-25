import cv2
import numpy as np
import motmetrics as mm
import csv
import time

# ==================== 0. 辅助函数 ====================
def load_gt_csv(gt_file_path):
    gt_data = {}
    try:
        with open(gt_file_path, 'r', newline='') as f:
            reader = csv.DictReader(f);
            for row in reader:
                frame_id = int(row['frame']);
                obj_id = int(row['id']);
                x = int(row['x']);
                y = int(row['y']);
                width = int(row['width']);
                height = int(row['height']);
                bbox = [x, y, width, height]
                if frame_id not in gt_data: gt_data[frame_id] = []
                gt_data[frame_id].append((obj_id, bbox))
    except FileNotFoundError:
        print(f"错误：找不到GT文件 '{gt_file_path}'"); return None
    except KeyError as e:
        print(f"错误：CSV缺少列: {e}"); return None
    return gt_data


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]);
    yA = max(boxA[1], boxB[1]);
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]);
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3]);
    interArea = max(0, xB - xA) * max(0, yB - yA);
    boxAArea = boxA[2] * boxA[3];
    boxBArea = boxB[2] * boxB[3]
    if boxAArea + boxBArea - interArea == 0: return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)


# ==================== 1. 改进的跟踪器类  =====================
class SpatialHistogramTracker:
    def __init__(self, feature_type='color', bins=12):
        self.feature_type = feature_type;
        self.bins = bins;
        self.model_hist = None;
        self.scale_factor = 2;
        self.spatial_sigma = 25.0 / self.scale_factor;
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def _get_features(self, frame):

        if self.feature_type == 'color':
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV); features = hsv[:, :, 0]; mask = cv2.inRange(hsv, np.array(
                (0., 60., 32.)), np.array((180., 255., 255.))); return features, mask
        # elif self.feature_type == 'edge':
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); otsu_thresh, _ = cv2.threshold(gray, 0, 255,
        #                                                                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU); gx = cv2.Sobel(
        #         gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1,
        #                                                          ksize=3); mag, ang = cv2.cartToPolar(gx, gy,
        #                                                                                               angleInDegrees=True); ang = np.mod(
        #         ang, 180); bool_mask = mag > otsu_thresh; mask = bool_mask.astype(
        #         np.uint8) * 255; features = ang;
            return features, mask

    def create_template(self, roi):

        scaled_roi = cv2.resize(roi, (roi.shape[1] // self.scale_factor, roi.shape[0] // self.scale_factor),
                                interpolation=cv2.INTER_AREA);
        h, w = scaled_roi.shape[:2];
        features, mask = self._get_features(scaled_roi);
        gauss_weights = cv2.getGaussianKernel(h, h / 4) * cv2.getGaussianKernel(w, w / 4).T;
        gauss_weights = (gauss_weights / gauss_weights.max());
        final_mask = cv2.bitwise_and(mask, mask, mask=(gauss_weights * 255).astype(np.uint8));
        raw_hist = [(0.0, 0.0, 0.0)] * self.bins;
        bin_width = 180.0 / self.bins;
        coords = np.where(final_mask > 0)
        for y, x in zip(*coords):
            bin_idx = int(features[y, x] / bin_width);
            if bin_idx >= self.bins: bin_idx = self.bins - 1;
            count, sum_x, sum_y = raw_hist[bin_idx];
            weight = gauss_weights[y, x];
            raw_hist[bin_idx] = (count + weight, sum_x + x * weight, sum_y + y * weight)
        self.model_hist = [];
        total_weight = sum(item[0] for item in raw_hist)
        if total_weight == 0: self.model_hist = [(0.0, np.array([w / 2, h / 2]))] * self.bins; return
        for count, sum_x, sum_y in raw_hist:
            if count > 0:
                norm_count = count / total_weight; mean_mu = np.array(
                    [sum_x / count, sum_y / count]); self.model_hist.append((norm_count, mean_mu))
            else:
                self.model_hist.append((0.0, np.array([w / 2, h / 2])))

    def _create_back_projection_map(self, frame, roi_bbox):

        x, y, w, h = [int(v) for v in roi_bbox];
        if w <= 0 or h <= 0: return np.zeros((1, 1), dtype=np.float32)
        roi = frame[y:y + h, x:x + w];
        features, mask = self._get_features(roi);
        if features is None: return np.zeros_like(roi[:, :, 0], dtype=np.float32) if len(
            roi.shape) > 2 else np.zeros_like(roi, dtype=np.float32)
        back_proj_map = np.zeros_like(features, dtype=np.float32);
        bin_width = 180.0 / self.bins;
        coords = np.where(mask > 0)
        for r, c in zip(*coords):
            bin_idx = int(features[r, c] / bin_width);
            if bin_idx >= self.bins: bin_idx = self.bins - 1;
            prob_feature = self.model_hist[bin_idx][0]
            if prob_feature > 0: mu_model = self.model_hist[bin_idx][1]; mu_cand = np.array([c, r]); dist_sq = np.sum(
                (mu_model - mu_cand) ** 2); prob_spatial = np.exp(-0.5 * dist_sq / (self.spatial_sigma ** 2));
            back_proj_map[r, c] = prob_feature * prob_spatial
        return back_proj_map

    def track(self, frame, search_window):
        scaled_frame = cv2.resize(frame, (frame.shape[1] // self.scale_factor, frame.shape[0] // self.scale_factor),
                                  interpolation=cv2.INTER_AREA);
        track_win = tuple(v // self.scale_factor for v in search_window)

        iters = 0;
        xc, yc = 0, 0
        for i in range(self.term_crit[1]):
            iters = i + 1;
            x, y, w, h = [int(v) for v in track_win];
            x = max(0, min(x, scaled_frame.shape[1] - 1));
            y = max(0, min(y, scaled_frame.shape[0] - 1));
            w = max(1, w);
            h = max(1, h);
            if x + w > scaled_frame.shape[1]: w = scaled_frame.shape[1] - x
            if y + h > scaled_frame.shape[0]: h = scaled_frame.shape[0] - y
            back_proj = self._create_back_projection_map(scaled_frame, (x, y, w, h));
            moments = cv2.moments(back_proj)
            if abs(moments['m00']) < 1e-5: break
            xc = int(moments['m10'] / moments['m00']);
            yc = int(moments['m01'] / moments['m00']);
            new_x = x + xc - w // 2;
            new_y = y + yc - h // 2;
            dist_moved = np.sqrt((new_x - x) ** 2 + (new_y - y) ** 2);
            track_win = (new_x, new_y, w, h)
            if dist_moved < self.term_crit[2]: break

        x, y, w, h = [int(v) for v in track_win];
        final_bbox_scaled = (x, y, w, h)

        back_proj_final = self._create_back_projection_map(scaled_frame, (x, y, w, h));
        coords = np.where(back_proj_final > 0)
        if len(coords[0]) > 10:
            x_coords, y_coords = coords[1], coords[0]
            new_w = int(np.std(x_coords) * 3 * 1.5)
            new_h = int(np.std(y_coords) * 3 * 1.5)
            w = max(10, new_w)
            h = max(10, new_h)
            final_bbox_scaled = (x + xc - w // 2, y + yc - h // 2, w, h)

        final_bbox = tuple(int(v * self.scale_factor) for v in final_bbox_scaled);
        final_center = (final_bbox[0] + final_bbox[2] / 2, final_bbox[1] + final_bbox[3] / 2);
        similarity_score = np.mean(
            back_proj_final) if 'back_proj_final' in locals() and back_proj_final.size > 0 else 0.0

        diag_info = {'iters': iters, 'new_w': w, 'new_h': h}
        return final_center, final_bbox, similarity_score, diag_info


# ==================== 2. 用户配置区和初始化 ====================
video_path = r"D:\PublicProject\HBY\zzy_all_data_GT_allFPS\video\a_01.mp4"
gt_file_path = r"D:\PublicProject\HBY\zzy_all_data_GT_allFPS\GT_allFPS_data\GT_a_01.csv"
window_name = 'Advanced Tracker with Diagnostics (v2)'
cap = cv2.VideoCapture(video_path)
gt_data = load_gt_csv(gt_file_path)
acc = mm.MOTAccumulator(auto_id=True)
tracker_color = SpatialHistogramTracker(feature_type='color')
# tracker_edge = SpatialHistogramTracker(feature_type='edge')
is_initialized = False
current_bbox = None

# 加入帧率统计初始化
frame_idx = 0
print_interval = 10
print_count = 0
start_time = time.time()

# ==================== 3. 跟踪与评测主循环 ====================
print("初始化完成，开始单次视频读取与评测...")
frame_idx = 0
print_interval = 10
print_count = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_idx += 1

    if not is_initialized and frame_idx in gt_data:
        bbox = gt_data[frame_idx][0][1]
        x0, y0, w0, h0 = [int(v) for v in bbox]
        if w0 > 0 and h0 > 0:
            initial_roi = frame[y0:y0 + h0, x0:x0 + w0]
            print(f"检测到第 {frame_idx} 帧的GT，正在初始化跟踪器...")
            tracker_color.create_template(initial_roi)
            # tracker_edge.create_template(initial_roi)
            current_bbox = bbox
            is_initialized = True
            print("初始化完成，开始跟踪。")
            continue

    if not is_initialized:
        vis = frame.copy()
        cv2.putText(vis, f"Frame: {frame_idx} (Searching for GT)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2)
        cv2.imshow(window_name, vis)
        if cv2.waitKey(1) & 0xFF == 27: break
        continue

    center_color, bbox_color, sim_color, diag_color = tracker_color.track(frame, current_bbox)
    # center_edge, bbox_edge, sim_edge, diag_edge = tracker_edge.track(frame, current_bbox)
    total_sim = sim_color# + sim_edge
    wc = sim_color / total_sim if total_sim > 1e-6 else 0.5
    we = 1.0 - wc
    fused_center_x = wc * center_color[0] #+ we * center_edge[0]
    fused_center_y = wc * center_color[1] #+ we * center_edge[1]
    fused_center = (int(fused_center_x), int(fused_center_y))

    # 使用平滑的方式更新尺寸
    prev_w, prev_h = current_bbox[2], current_bbox[3]
    fused_w = wc * bbox_color[2] #+ we * bbox_edge[2]
    fused_h = wc * bbox_color[3] #+ we * bbox_edge[3]
    # 学习率，0.1表示新尺寸占10%权重，旧尺寸占90%权重，防止剧烈变化
    learning_rate = 0.1
    new_w = int(prev_w * (1 - learning_rate) + fused_w * learning_rate)
    new_h = int(prev_h * (1 - learning_rate) + fused_h * learning_rate)

    current_bbox = (int(fused_center[0] - new_w / 2), int(fused_center[1] - new_h / 2), new_w, new_h)

    final_pred_box = list(current_bbox)

    if frame_idx in gt_data:
        gt_items = gt_data[frame_idx]
        gt_ids = [item[0] for item in gt_items]
        gt_bboxes = [item[1] for item in gt_items]
        pred_ids = [1]
        pred_bboxes = [final_pred_box]
        distances = mm.distances.iou_matrix(gt_bboxes, pred_bboxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances)

        print_count += 1

    vis = frame.copy()
    (x, y, w, h) = final_pred_box
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(vis, 'Prediction', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if frame_idx in gt_data:
        for _, gt_bbox in gt_data[frame_idx]:
            (x_gt, y_gt, w_gt, h_gt) = gt_bbox
            cv2.rectangle(vis, (x_gt, y_gt), (x_gt + w_gt, y_gt + h_gt), (0, 255, 0), 2)
            cv2.putText(vis, 'GroundTruth', (x_gt, y_gt - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vis, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow(window_name, vis)

    if cv2.waitKey(1) & 0xFF == 27: break

# ==================== 4. 计算并显示结果 ====================
print("\n视频播放完毕，正在计算评测指标...")
cap.release()
cv2.destroyAllWindows()

end_time = time.time()
elapsed_time = end_time - start_time
avg_fps = frame_idx / elapsed_time if elapsed_time > 0 else 0
print(f"平均帧率（Average FPS）: {avg_fps:.2f}")

mh = mm.metrics.create()
summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='overall')
strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
print(strsummary)
