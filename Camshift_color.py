import cv2
import numpy as np
import time
# ==================== 0. 性能优化配置 ====================
# 将此值设置为 > 1 的整数以缩小图像进行处理，例如 2 表示尺寸减半
# 值越大，速度越快，但精度可能下降
PERFORMANCE_SCALE_FACTOR = 2 

# ==================== 1. 改进的跟踪器类 ====================
class SpatialHistogramTracker:
    """
    一个实现了论文核心思想的改进跟踪器类。
    它不再使用OpenCV的CamShift，而是手动实现了更复杂的特征提取和匹配逻辑，
    以改进后更精确的方法达到目标跟踪的目的。
    
    主要实现了以下论文要点：
    1. 【二阶空间直方图】: 不仅统计特征值，还计算其空间均值位置 μ。
    2. 【中心距离加权】: 在创建模板时，赋予靠近中心的像素更高的权重。
    3. 【综合相似度度量】: 结合特征分布和空间分布来评估相似度。
    """
    def __init__(self, feature_type='color', bins=12):
        self.feature_type = feature_type
        self.bins = bins
        self.model_hist = None
        # 空间惩罚参数。值越小，对空间位置偏移的惩罚越强。
        self.spatial_sigma = 25.0 / PERFORMANCE_SCALE_FACTOR

    def _get_features(self, frame):
        """
        根据指定的特征类型，从图像帧中提取特征通道和对应的有效像素掩码。
        这是“颜色空间分布直方图”的数据来源。
        """
        if self.feature_type == 'color':
            # 对应论文中的“颜色空间分布直方图”的特征提取
            # 使用HSV颜色空间，因其对光照变化更鲁棒
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # 使用色调(Hue)通道作为主要特征
            features = hsv[:, :, 0]
            # 过滤掉低饱和度的像素（如黑、白、灰）
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            return features, mask
        return None, None

    def create_template(self, roi):
        """从初始ROI创建特征模板，即目标的“空间直方图模型”。"""
        scaled_roi = cv2.resize(
            roi, 
            (roi.shape[1] // PERFORMANCE_SCALE_FACTOR, roi.shape[0] // PERFORMANCE_SCALE_FACTOR), 
            interpolation=cv2.INTER_AREA
        )
        h, w = scaled_roi.shape[:2]
        features, mask = self._get_features(scaled_roi)
        
        # 【论文要点】实现中心距离加权 (对应论文公式 7 的 λ(d) 函数思想)
        # 创建一个2D高斯权重图，中心权重为1，向边缘衰减
        gauss_weights = cv2.getGaussianKernel(h, h/4) * cv2.getGaussianKernel(w, w/4).T
        gauss_weights = (gauss_weights / gauss_weights.max())
        final_mask = cv2.bitwise_and(mask, mask, mask=(gauss_weights * 255).astype(np.uint8))

        # 【论文要点】实现二阶空间直方图，累加坐标以计算每个bin的空间均值 μ (对应论文公式 2, 9)
        # 存储结构: (加权像素数, 加权x坐标总和, 加权y坐标总和)
        raw_hist = [(0.0, 0.0, 0.0)] * self.bins
        bin_width = 180.0 / self.bins
        coords = np.where(final_mask > 0)
        for y, x in zip(*coords):
            bin_idx = int(features[y, x] / bin_width)
            if bin_idx >= self.bins: bin_idx = self.bins - 1
            count, sum_x, sum_y = raw_hist[bin_idx]
            weight = gauss_weights[y, x] # 使用高斯权重
            raw_hist[bin_idx] = (count + weight, sum_x + x * weight, sum_y + y * weight)

        self.model_hist = []
        total_weight = sum(item[0] for item in raw_hist)
        if total_weight == 0:
            print(f"警告: 未能为 '{self.feature_type}' 特征找到有效像素。")
            self.model_hist = [(0.0, np.array([w/2, h/2]))] * self.bins
            return
            
        # 计算最终的模板: (归一化计数值, 均值向量μ)
        for count, sum_x, sum_y in raw_hist:
            if count > 0:
                norm_count = count / total_weight
                mean_mu = np.array([sum_x / count, sum_y / count])
                self.model_hist.append((norm_count, mean_mu))
            else:
                self.model_hist.append((0.0, np.array([w/2, h/2])))

    def _calculate_similarity(self, candidate_roi):
        """
        【论文要点】实现一个综合相似度度量 (对应论文公式 11, 12, 13 的 ρ(y) 思想)。
        该度量结合了特征分布相似度（颜色/边缘）和空间分布相似度（μ的位置）。
        """
        if self.model_hist is None: return 0.0
        h, w = candidate_roi.shape[:2]
        features, mask = self._get_features(candidate_roi)
        raw_hist_cand = [(0, 0.0, 0.0)] * self.bins
        bin_width = 180.0 / self.bins
        coords = np.where(mask > 0)
        for y, x in zip(*coords):
            bin_idx = int(features[y, x] / bin_width)
            if bin_idx >= self.bins: bin_idx = self.bins - 1
            count, sum_x, sum_y = raw_hist_cand[bin_idx]
            raw_hist_cand[bin_idx] = (count + 1, sum_x + x, sum_y + y)
        total_pixels_cand = sum(item[0] for item in raw_hist_cand)
        if total_pixels_cand == 0: return 0.0
        
        h_model = np.array([item[0] for item in self.model_hist])
        h_cand_counts = np.array([item[0] for item in raw_hist_cand])
        h_cand_norm = h_cand_counts / total_pixels_cand
        
        # 1. 计算特征分布相似度 (使用巴氏系数)
        feature_similarity = np.sum(np.sqrt(h_model * h_cand_norm))
        
        # 2. 计算加权的空间分布相似度
        spatial_sim_sum = 0.0
        for i in range(self.bins):
            if h_cand_counts[i] > 0:
                _, mu_model = self.model_hist[i]
                _, sum_x, sum_y = raw_hist_cand[i]
                mu_cand = np.array([sum_x / h_cand_counts[i], sum_y / h_cand_counts[i]])
                # 使用高斯函数惩罚候选区域μ与模型μ之间的空间距离
                dist_sq = np.sum((mu_model - mu_cand)**2)
                spatial_penalty = np.exp(-0.5 * dist_sq / (self.spatial_sigma**2))
                # 综合考虑该bin的特征相似度和空间相似度
                spatial_sim_sum += np.sqrt(h_model[i] * h_cand_norm[i]) * spatial_penalty
                
        # 最终相似度是两者的乘积，综合反映了特征和空间的匹配程度
        return feature_similarity * spatial_sim_sum

    def track(self, frame, search_window):
        """执行跟踪，找到与模型最相似的图像块。"""
        scaled_frame = cv2.resize(
            frame, 
            (frame.shape[1] // PERFORMANCE_SCALE_FACTOR, frame.shape[0] // PERFORMANCE_SCALE_FACTOR),
            interpolation=cv2.INTER_AREA
        )
        scaled_window = [v // PERFORMANCE_SCALE_FACTOR for v in search_window]
        x, y, w, h = scaled_window
        best_sim = -1.0
        best_pos = (x, y)
        step = int(max(w, h) * 0.1) if max(w,h) > 10 else 1
        # 简化的5点网格搜索，代替完整的MeanShift迭代，以提高性能
        search_positions = [(0, 0), (0, -step), (0, step), (-step, 0), (step, 0)]
        for dx, dy in search_positions:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx + w >= scaled_frame.shape[1] or ny + h >= scaled_frame.shape[0]:
                continue
            candidate_roi = scaled_frame[ny:ny+h, nx:nx+w]
            sim = self._calculate_similarity(candidate_roi)
            if sim > best_sim:
                best_sim = sim
                best_pos = (nx, ny)
        final_bbox_scaled = (best_pos[0], best_pos[1], w, h)
        final_bbox = [v * PERFORMANCE_SCALE_FACTOR for v in final_bbox_scaled]
        final_center = (final_bbox[0] + final_bbox[2]/2, final_bbox[1] + final_bbox[3]/2)
        return final_center, final_bbox, best_sim

# ==================== 2. 初始化 ====================
video_path = r"D:\PublicProject\HBY\zzy_Camshift\zzy_camshift_data\final_video\final_01.mp4"
window_name = 'Optimized Multi-feature Tracker'
cap = cv2.VideoCapture(video_path)
ret, first_frame = cap.read()
bbox = cv2.selectROI("Select ROI", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select ROI")
x0, y0, w0, h0 = [int(v) for v in bbox]
initial_roi = first_frame[y0:y0+h0, x0:x0+w0]

# 【论文要点】多特征融合策略：创建两个独立的跟踪器实例，分别处理颜色和边缘特征
tracker_color = SpatialHistogramTracker(feature_type='color')
tracker_edge = SpatialHistogramTracker(feature_type='edge')

print("正在创建特征模板...")
tracker_color.create_template(initial_roi)


current_bbox = bbox
print("初始化完成，开始跟踪...")

# ==================== 3. 跟踪主循环 ====================
prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret: break

    center_color, bbox_color, sim_color = tracker_color.track(frame, current_bbox)
    total_sim = sim_color
    wc = sim_color / total_sim if total_sim > 1e-6 else 0.5
    we = 1.0 - wc
    # c) 【论文要点】根据权重计算最终融合后的中心点 (对应论文公式 21)
    fused_center_x = wc * center_color[0]
    fused_center_y = wc * center_color[1]
    fused_center = (int(fused_center_x), int(fused_center_y))
    # d) 【优化】耦合跟踪器：使用融合后的结果来统一更新下一帧的搜索窗口，防止单个跟踪器漂移
    current_bbox = (int(fused_center[0] - w0/2), int(fused_center[1] - h0/2), w0, h0)

    #  实时帧率计算和显示
    now = time.time()
    fps = 1.0 / (now - prev_time)
    prev_time = now

    vis = frame.copy()
    (x,y,w,h) = [int(v) for v in bbox_color]
    cv2.rectangle(vis, (x,y), (x+w,y+h), (255,0,0), 2)
    cv2.putText(vis, f'Color Sim: {sim_color:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
    cv2.rectangle(vis, (x,y), (x+w,y+h), (0,0,255), 2)
    cv2.putText(vis, f'FPS: {fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)  # 左上角帧率

    cv2.imshow(window_name, vis)
    if cv2.waitKey(1) & 0xFF == 27: break

# ==================== 4. 清理资源 ====================
cap.release()
cv2.destroyAllWindows()