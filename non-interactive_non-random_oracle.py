import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import hashlib
        
def moving_average(data, window_size=7):
    """Smooth data with a simple moving average."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

class OscilloscopeDisplay:
    def __init__(self, width=800, height=400, history_length=50):
        self.width = width
        self.height = height
        self.history = deque(maxlen=history_length)
        self.time_axis = np.linspace(0, 2*np.pi, width)
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.grid(True, color='#00FF00', alpha=0.3, linestyle='-', linewidth=0.5)
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(-2, 2)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.phosphor_green = '#00FF00'
        self.dim_green = '#004400'

    def add_waveform(self, data):
        if len(data) == 0:
            return
        if np.max(np.abs(data)) > 0:
            normalized_data = data / np.max(np.abs(data)) * 1.5
        else:
            normalized_data = data
        if len(normalized_data) != self.width:
            x_old = np.geomspace(1, self.width-1, len(normalized_data))
            x_new = np.geomspace(1, self.width-1, self.width)
            normalized_data = np.interp(x_new, x_old, normalized_data)
        self.history.append(normalized_data)

    def render(self):
        self.ax.clear()
        self.ax.set_facecolor('black')
        self.ax.grid(True, color=self.dim_green, alpha=0.3, linestyle='-', linewidth=0.5)
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(-2, 2)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.axhline(y=0, color=self.dim_green, linewidth=1, alpha=0.5)
        if len(self.history) > 1:
            for i, trace in enumerate(list(self.history)[:-1]):
                alpha = 0.1 + (i / len(self.history)) * 0.3
                self.ax.plot(range(self.width), trace, color=self.phosphor_green, alpha=alpha, linewidth=1)
        if len(self.history) > 0:
            current_trace = self.history[-1]
            for i in range(3):
                self.ax.plot(range(self.width), current_trace, color=self.phosphor_green, alpha=0.2-i*0.05, linewidth=6+i*2)
            self.ax.plot(range(self.width), current_trace, color=self.phosphor_green, alpha=1.0, linewidth=2)
        self.ax.text(0.02, 0.95, 'CURVATURE OSCILLOSCOPE', transform=self.ax.transAxes, color=self.phosphor_green,
                     fontsize=12, fontweight='bold', family='monospace')
        if len(self.history) > 0:
            current_data = self.history[-1]
            peak_val = np.max(np.abs(current_data))
            rms_val = np.sqrt(np.mean(current_data**2))
            self.ax.text(0.02, 0.05, f'PEAK: {peak_val:.3f}', transform=self.ax.transAxes, color=self.phosphor_green,
                         fontsize=10, family='monospace')
            self.ax.text(0.02, 0.12, f'RMS:  {rms_val:.3f}', transform=self.ax.transAxes, color=self.phosphor_green,
                         fontsize=10, family='monospace')
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.fig.canvas.draw()
        try:
            buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = self.fig.canvas.get_width_height()
            scope_img = buf.reshape(h, w, 4)[:, :, :3]
        except AttributeError:
            try:
                buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                w, h = self.fig.canvas.get_width_height()
                scope_img = buf.reshape(h, w, 3)
            except AttributeError:
                buf = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
                w, h = self.fig.canvas.get_width_height()
                scope_img = buf.reshape(h, w, 4)[:, :, 1:4]
        scope_img = self.add_scanlines(scope_img)
        return scope_img

    def add_scanlines(self, img):
        h, w = img.shape[:2]
        for y in range(0, h, 3):
            if y < h:
                img[y, :] = img[y, :] * 0.8
        return img

class CurvatureSmoother:
    def __init__(self, maxlen=5):
        self.history = deque(maxlen=maxlen)
    def add(self, curv):
        self.history.append(curv)
    def get_smoothed(self):
        if not self.history:
            return []
        max_len = max(len(c) for c in self.history)
        padded = [np.pad(c, (0, max_len - len(c)), 'constant') for c in self.history]
        avg = np.mean(padded, axis=0)
        return avg

def curvature(contour, k=15):
    n = len(contour)
    curvatures = []
    for i in range(n):
        p1 = contour[i % n][0]
        p2 = contour[(i - k) % n][0]
        p3 = contour[(i + k) % n][0]
        mat = np.array([
            [p1[0], p1[1], 1],
            [p2[0], p2[1], 1],
            [p3[0], p3[1], 1]
        ])
        A = np.linalg.det(mat)
        if abs(A) < 1e-6:
            curvatures.append(0)
            continue
        Dx = np.linalg.det(np.array([
            [p1[0]**2 + p1[1]**2, p1[1], 1],
            [p2[0]**2 + p2[1]**2, p2[1], 1],
            [p3[0]**2 + p3[1]**2, p3[1], 1]
        ]))
        Dy = -np.linalg.det(np.array([
            [p1[0]**2 + p1[1]**2, p1[0], 1],
            [p2[0]**2 + p2[1]**2, p2[0], 1],
            [p3[0]**2 + p3[1]**2, p3[0], 1]
        ]))
        xc = Dx / (2*A)
        yc = Dy / (2*A)
        r = np.sqrt((p1[0] - xc)**2 + (p1[1] - yc)**2)
        curvature_val = 1/r if r != 0 else 0
        curvatures.append(curvature_val)
    return np.array(curvatures)
def rate_of_change_sign_flips(data):
    diffs = np.diff(data)
    signs = np.sign(diffs)
    sign_flips = np.diff(signs)
    flips_indices = np.where(sign_flips != 0)[0]
    if len(flips_indices) < 2:
        return []
    intervals = np.diff(flips_indices)
    rate_of_change = intervals[1:] / intervals[:-1]
    return rate_of_change
# Main execution
cap = cv2.VideoCapture(0)
oscilloscope = OscilloscopeDisplay(width=800, height=400, history_length=20)
smoother = CurvatureSmoother(maxlen=5)

print("Starting Oscilloscope-Style Green Knot Analyzer...")
print("Press ESC to quit")
STATE_SPACE = 99999999999
for cost in range(STATE_SPACE):
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([170, 70, 50])
    upper_red = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    display_frame = frame.copy()
    knot_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if area > 50 and circularity > 0.7:
            knot_count += 1
            if len(cnt) >= 10:
                
               
                curv = curvature(cnt, k=15)
                curv_smooth = moving_average(curv, window_size=9)
                smoother.add(curv_smooth[(len(curv_smooth)//3)*1:(len(curv_smooth)//3)*2])
                curv_time_smooth = smoother.get_smoothed()
                oscilloscope.add_waveform(abs(-cost/curv_time_smooth))
                cv2.drawContours(display_frame, [cnt], -1, (0, 0, 255), 2)
                M = cv2.moments(cnt)
                arr = np.array(smoother.get_smoothed())
                window = arr[-15:] if len(arr) >= 15 else arr  # last 15 values for better pattern
                diffs = np.diff(window)  # consecutive differences
                signs = np.sign(diffs)
                sign_flips = np.diff(signs)
                
                # A sign flip is where sign_flips != 0
                num_flips = np.sum(sign_flips != 0)
                # Check if any difference exceeds 0.1
                if cost**2 < STATE_SPACE and num_flips > 1:
                    # Your action here
                    print(len(rate_of_change_sign_flips(sign_flips))*sum(rate_of_change_sign_flips(sign_flips))*sign_flips)
                    

                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(display_frame, (cx, cy), 5, (0, 255, 255), -1)
                    cv2.putText(display_frame, f'Knot: Area={area:.0f}, Circ={circularity:.2f}',
                                (cx-40, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(display_frame, f'Knots detected: {knot_count}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    scope_display = oscilloscope.render()
    cv2.imshow('Green Knot Detection', display_frame)
    cv2.imshow('Curvature Oscilloscope', scope_display)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
plt.close('all')
print("Oscilloscope session ended.")