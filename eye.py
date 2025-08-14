import pygame
import cv2
import math
import os
import sys
import numpy as np

class FaceTrackingEye:
    def __init__(self, width=640, height=240, fps=30):
        self.width = width  
        self.height = height
        self.feed_width = width // 2 
        self.feed_height = height
        self.fps = fps
        self.init_pygame()
        self.init_opencv()
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)  
        
        # Eye properties (First Eye - Left)
        self.eye_radius = 80
        self.pupil_radius = 30
        self.pupil_max_movement = 40
        self.eye_center = (self.feed_width + self.feed_width // 4, self.height // 2)  # x=400, y=120
        
        # Second Eye properties (Right)
        self.eye_center2 = (self.feed_width + 3 * self.feed_width // 4, self.height // 2)  # x=560, y=120
        
        # Face tracking variables
        self.last_face_x = self.feed_width // 2
        self.last_face_y = self.feed_height // 2
        self.face_detected = False
        self.no_face_counter = 0
        self.RESET_DELAY = 3 * self.fps  
        self.smoothing_alpha = 0.3  # Reduced for more responsiveness
        
        # Dynamic face box size for scaling
        self.face_box_width = self.feed_width // 2  
        self.face_box_height = self.feed_height // 2  
        
        # Font setup for "दृष्टि"
        font_names = ["mangal", "arialunicode", "sans-serif"]
        self.font = None
        for font_name in font_names:
            try:
                self.font = pygame.font.SysFont(font_name, 30)
                break
            except:
                continue
        if self.font is None:
            print("Warning: Could not load specified fonts. Using default font.")
            self.font = pygame.font.Font(None, 30)  
        self.title_text = "VISION"
        
        # Debug font
        pygame.font.init()
        self.debug_font = pygame.font.SysFont("monospace", 15)
        
    def init_pygame(self):
        """Initialize Pygame and create window."""
        pygame.init()
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Face Tracking Eyes")
        self.clock = pygame.time.Clock()
        
    def init_opencv(self):
        """Initialize OpenCV webcam and face cascade."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            sys.exit(1)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.feed_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.feed_height)
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            print("Error: Haar Cascade file not found")
            sys.exit(1)
            
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print("Error: Could not load Haar Cascade classifier")
            sys.exit(1)
            
    def get_pupil_position(self, target_x, target_y):
        """Calculate pupil positions for both eyes with accurate movement matching."""
        # Feed center for reference
        feed_center_x = self.feed_width // 2
        feed_center_y = self.feed_height // 2
        
        # Normalize face coordinates based on dynamic face box size
        norm_x = (target_x - feed_center_x) / (self.face_box_width // 2)  # -1 to 1 within face range
        norm_y = (target_y - feed_center_y) / (self.face_box_height // 2)  # -1 to 1 within face range
        
        # Clamp normalized values to [-1, 1] to handle edge cases
        norm_x = max(min(norm_x, 1.0), -1.0)
        norm_y = max(min(norm_y, 1.0), -1.0)
        
        # Scale to pupil movement range
        dx = norm_x * self.pupil_max_movement
        dy = norm_y * self.pupil_max_movement
        
        # Clamp to ensure pupil stays within bounds
        distance = math.sqrt(dx**2 + dy**2)
        if distance > self.pupil_max_movement:
            dx = dx * self.pupil_max_movement / distance
            dy = dy * self.pupil_max_movement / distance
            
        # Apply to first eye center (left eye)
        pupil_x = int(self.eye_center[0] + dx)
        pupil_y = int(self.eye_center[1] + dy)
        
        # Apply to second eye center (right eye)
        pupil_x2 = int(self.eye_center2[0] + dx)
        pupil_y2 = int(self.eye_center2[1] + dy)
        
        return (pupil_x, pupil_y), (pupil_x2, pupil_y2), (norm_x, norm_y)
    
    def process_frame(self):
        """Capture and process webcam frame for face detection."""
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame")
            return None
            
        frame = cv2.flip(frame, 1)  # Mirror frame
        frame = cv2.resize(frame, (self.feed_width, self.feed_height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        if len(faces) > 0:
            self.face_detected = True
            self.no_face_counter = 0
            
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            (x, y, w, h) = largest_face
            
            # Update dynamic face box size
            self.face_box_width = w
            self.face_box_height = h
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.GREEN, 3)
            
            # Calculate tracking point between the eyes
            face_center_x = x + w // 2
            face_center_y = y + h // 3  # Between the eyes
            
            # Draw crosshair
            cv2.line(frame, (face_center_x - 10, face_center_y), 
                    (face_center_x + 10, face_center_y), self.RED, 2)
            cv2.line(frame, (face_center_x, face_center_y - 10), 
                    (face_center_x, face_center_y + 10), self.RED, 2)
            
            # Smooth the tracking point
            self.last_face_x = self.smoothing_alpha * face_center_x + (1 - self.smoothing_alpha) * self.last_face_x
            self.last_face_y = self.smoothing_alpha * face_center_y + (1 - self.smoothing_alpha) * self.last_face_y
        else:
            self.no_face_counter += 1
            if self.no_face_counter > self.RESET_DELAY:
                self.face_detected = False
                self.last_face_x = self.feed_width // 2
                self.last_face_y = self.feed_height // 2
                
        return frame
    
    def render(self, frame):
        """Render the webcam feed, eyes, and title in Pygame window."""
        self.screen.fill(self.BLACK)
        
        # Render webcam feed (left half)
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = np.rot90(frame_rgb)
            frame_surface = pygame.surfarray.make_surface(frame_rgb)
            self.screen.blit(frame_surface, (0, 0))
        
        # Draw separator line
        pygame.draw.line(self.screen, self.WHITE, (self.feed_width, 0), 
                        (self.feed_width, self.height), 2)
        
        # Render eyes (right half)
        pupil1, pupil2, (norm_x, norm_y) = self.get_pupil_position(self.last_face_x, self.last_face_y)
        eye_color = self.GREEN if self.face_detected else self.WHITE
        
        # First eye (left)
        pygame.draw.circle(self.screen, eye_color, self.eye_center, self.eye_radius)
        pygame.draw.circle(self.screen, self.BLACK, self.eye_center, self.eye_radius, 3)
        pygame.draw.circle(self.screen, self.BLACK, pupil1, self.pupil_radius)
        
        # Second eye (right)
        pygame.draw.circle(self.screen, eye_color, self.eye_center2, self.eye_radius)
        pygame.draw.circle(self.screen, self.BLACK, self.eye_center2, self.eye_radius, 3)
        pygame.draw.circle(self.screen, self.BLACK, pupil2, self.pupil_radius)
        
        # Render title "दृष्टि" at top center with shadow
        shadow_surface = self.font.render(self.title_text, True, self.BLACK)
        shadow_rect = shadow_surface.get_rect(center=(self.width // 2 + 2, 22))
        self.screen.blit(shadow_surface, shadow_rect)
        text_surface = self.font.render(self.title_text, True, self.YELLOW)
        text_rect = text_surface.get_rect(center=(self.width // 2, 20))
        self.screen.blit(text_surface, text_rect)
        
        # Debug info: Display normalized coordinates
        debug_text = f"Norm X: {norm_x:.2f}, Norm Y: {norm_y:.2f}"
        debug_surface = self.debug_font.render(debug_text, True, self.WHITE)
        self.screen.blit(debug_surface, (self.feed_width + 10, 10))
        
    def run(self):
        """Main game loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                        print("Program terminated by 'q' key press")
                    
            frame = self.process_frame()
            if frame is None:
                break
                
            self.render(frame)
            pygame.display.flip()
            self.clock.tick(self.fps)
            
    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        
def main():
    try:
        app = FaceTrackingEye()
        app.run()
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()