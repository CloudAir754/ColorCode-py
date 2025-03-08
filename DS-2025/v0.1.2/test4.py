import cv2
import numpy as np

class ColorCodeDetector:
    def __init__(self, image_path):
        self.orig = cv2.imread(image_path)
        if self.orig is None:
            raise ValueError("Image load failed")
        
        self.process = []  # Store processing steps for visualization
        self.target_size = 800
        self.min_contour_area = 500

    # Visualization control
    def _add_step(self, img, title):
        """统一处理图像为三通道后再保存"""
        # 转换单通道图像为三通道
        if len(img.shape) == 2:
            display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            display = img.copy()
        
        # 调整尺寸并添加标题
        display = cv2.resize(display, (400, 400))
        cv2.putText(display, title, (10,30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        self.process.append(display)

    # Image preprocessing
    def _preprocess(self):
        """Image preprocessing pipeline"""
        # Resize
        img = cv2.resize(self.orig, (self.target_size, self.target_size))
        self._add_step(img, "1. Original")
        
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._add_step(gray, "2. Grayscale")
        
        # Histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized = clahe.apply(gray)
        self._add_step(equalized, "3. Equalized")
        
        # Edge detection
        blurred = cv2.GaussianBlur(equalized, (5,5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        self._add_step(edged, "4. Canny Edges")
        
        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        self._add_step(closed, "5. Morphology")
        
        return closed

    # Perspective transformation
    def _warp_perspective(self, contour):
        """Perform perspective transformation"""
        # Order points: tl, tr, br, bl
        pts = contour.reshape(4,2)
        rect = np.zeros((4,2), dtype=np.float32)
        
        # Sort by sum and difference
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]  # Top-right
        rect[3] = pts[np.argmax(d)]  # Bottom-left

        # Calculate target dimensions
        (tl, tr, br, bl) = rect
        width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
        height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
        
        # Destination points
        dst = np.array([
            [0, 0],
            [width-1, 0],
            [width-1, height-1],
            [0, height-1]], dtype=np.float32)

        # Perspective transform
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.orig, M, (int(width), int(height)))
        return warped

    # Color recognition
    def _detect_colors(self, warped):
        """Detect colors in 3x3 grid"""
        h, w = warped.shape[:2]
        cell_size = min(h, w) // 3
        colors = []
        
        # Create a copy of the warped image to draw annotations
        annotated_image = warped.copy()
        
        for row in range(3):
            for col in range(3):
                # Calculate cell coordinates
                y1 = row * cell_size + cell_size//4
                y2 = y1 + cell_size//2
                x1 = col * cell_size + cell_size//4
                x2 = x1 + cell_size//2
                
                # Extract cell region
                cell = warped[y1:y2, x1:x2]
                
                # Convert to HSV
                hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
                mean_hsv = np.mean(hsv, axis=(0,1))
                
                # Classify color
                color = self._classify_color(mean_hsv) # 这里是返回对应区域的颜色
                colors.append(color)
                
                # Annotate the color on the image
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.putText(annotated_image, color, (center_x - 20, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Format as 3x3 matrix
        return [colors[i*3:(i+1)*3] for i in range(3)], annotated_image

    def _classify_color(self, hsv):
        """Classify color based on HSV values"""
        hue, sat, val = hsv
        hue *= 2  # Convert to 0-360 range
        
        if val < 50: return 'Black'
        if sat < 50 and val > 200: return 'White'
        if (0 <= hue <= 15) or (165 <= hue <= 180): return 'Red'
        if 40 <= hue <= 80: return 'Green'
        if 100 <= hue <= 140: return 'Blue'
        return 'Unknown'

    # Main process
    def analyze(self):
        """Main analysis pipeline"""
        try:
            # Step 1: Preprocessing
            processed = self._preprocess()
            
            # Step 2: Find contours
            contours, _ = cv2.findContours(processed,
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours
            candidates = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_contour_area: continue
                
                # Approximate polygon
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
                if len(approx) == 4:
                    candidates.append(approx)
            
            if not candidates:
                raise ValueError("No 3D code found")
            
            # Select largest contour
            largest = max(candidates, key=cv2.contourArea)
            
            # Step 3: Perspective transform
            warped = self._warp_perspective(largest)
            self._add_step(warped, "6. Warped")
            
            # Step 4: Color detection
            color_matrix, annotated_image = self._detect_colors(warped)
            self._add_step(annotated_image, "7. Annotated Colors")
            
            # Show each processing step in separate windows
            for i, step in enumerate(self.process):
                cv2.imshow(f'Step {i+1}', step)
            
            return {
                'status': 'success',
                'colors': color_matrix,
                'warped': warped,
                'annotated': annotated_image
            }
            
        except Exception as e:
            # Create error display
            err_img = np.zeros((400,400,3), dtype=np.uint8)
            cv2.putText(err_img, "3D Code Not Found", (50,200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            self.process.append(err_img)
            
            # Show each processing step in separate windows
            for i, step in enumerate(self.process):
                cv2.imshow(f'Step {i+1}', step)
            
            return {
                'status': 'error',
                'message': str(e)
            }

if __name__ == "__main__":
    detector = ColorCodeDetector('./Sample/Pic2-2.jpg')
    result = detector.analyze()
    
    if result['status'] == 'success':
        print("Color Matrix:")
        for row in result['colors']:
            print(row)
        cv2.imshow('Warped Output', result['warped'])
        cv2.imshow('Annotated Colors', result['annotated'])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()