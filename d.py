import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime
from gtts import gTTS
import pygame
import math

def speak(text, lang='hi', speed=2.5):
    print(f"Speaking: {text}")  
    tts = gTTS(text=text, lang=lang)
    tts.save("temp.mp3")
    os.system(f"ffplay -nodisp -autoexit -af atempo={speed} temp.mp3")
    os.remove("temp.mp3")

# Dlib setup
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/cpatwadityasharma/attendence/Face-Recognition-Based-Attendance-System/data/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("/home/cpatwadityasharma/attendence/Face-Recognition-Based-Attendance-System/data/dlib_face_recognition_resnet_model_v1.dat")

# Database setup
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()
current_date = datetime.datetime.now().strftime("%Y_%m_%d")
table_name = "attendance"
create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (name TEXT, time TEXT, date DATE, UNIQUE(name, date))"
cursor.execute(create_table_sql)
conn.commit()
conn.close()

class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # Frame counter
        self.frame_cnt = 0

        # Face database lists
        self.face_features_known_list = []
        self.face_name_known_list = []

        # Centroid tracking
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0
        self.current_frame_face_X_e_distance_list = []
        self.current_frame_face_position_list = []
        self.current_frame_face_feature_list = []
        self.last_current_frame_centroid_e_distance = 0
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

        # Pygame eye tracking setup
        self.pygame_width = 320
        self.pygame_height = 240
        self.feed_width = 320
        self.feed_height = 240
        self.fps_target = 30
        self.init_pygame()

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)

        # Eye properties
        self.eye_radius = 80
        self.pupil_radius = 30
        self.pupil_max_movement = 40
        self.eye_center = (self.pygame_width // 2, self.pygame_height // 2)

        # Face tracking variables for eye movement
        self.last_face_x = self.feed_width // 2
        self.last_face_y = self.feed_height // 2
        self.face_detected = False
        self.no_face_counter = 0
        self.RESET_DELAY = 3 * self.fps_target
        self.smoothing_alpha = 0.3

        # Debug font for Pygame
        pygame.font.init()
        self.debug_font = pygame.font.SysFont("monospace", 15)

        # Font setup for title
        font_names = ["mangal", "arialunicode", "sans-serif"]
        self.font_pygame = None
        for font_name in font_names:
            try:
                self.font_pygame = pygame.font.SysFont(font_name, 30)
                break
            except:
                continue
        if self.font_pygame is None:
            print("Warning: Could not load specified fonts. Using default font.")
            self.font_pygame = pygame.font.Font(None, 30)
        self.title_text = "दृष्टि"

    def init_pygame(self):
        """Initialize Pygame and create window."""
        pygame.init()
        os.environ['SDL_VIDEO_WINDOW_POS'] = "320,0"
        self.screen = pygame.display.set_mode((self.pygame_width, self.pygame_height))
        pygame.display.set_caption("Eye Tracker")
        self.clock = pygame.time.Clock()

    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            return 0

    def update_fps(self):
        now = time.time()
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time if self.frame_time > 0 else 0
        self.frame_start_time = now

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])
                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)
            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    def draw_note(self, img_rd):
        cv2.putText(img_rd, "Face Recognizer with Deep Learning", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 210), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple(
                [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
                                 self.font, 0.8, (255, 190, 0), 1, cv2.LINE_AA)

    def attendance(self, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
        existing_entry = cursor.fetchone()

        if existing_entry:
            print(f"{name} is already marked as present for {current_date}")
        else:
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            cursor.execute("INSERT INTO attendance (name, time, date) VALUES (?, ?, ?)", (name, current_time, current_date))
            conn.commit()
            print(f"{name} marked as present for {current_date} at {current_time}")
        conn.close()

    def get_pupil_position(self, target_x, target_y):
        """Calculate pupil position with accurate movement matching."""
        # Normalize face coordinates relative to the feed's center (160, 120)
        feed_center_x = self.feed_width // 2  # 160
        feed_center_y = self.feed_height // 2  # 120

        # Normalize to [-1, 1] based on the feed dimensions
        norm_x = (target_x - feed_center_x) / feed_center_x  # -1 (left) to 1 (right)
        norm_y = (target_y - feed_center_y) / feed_center_y  # -1 (top) to 1 (bottom)

        # Scale to pupil movement range
        dx = norm_x * self.pupil_max_movement
        dy = norm_y * self.pupil_max_movement

        # Clamp to ensure pupil stays within bounds
        distance = math.sqrt(dx**2 + dy**2)
        if distance > self.pupil_max_movement:
            dx = dx * self.pupil_max_movement / distance
            dy = dy * self.pupil_max_movement / distance

        # Apply to eye center in Pygame window
        pupil_x = int(self.eye_center[0] + dx)
        pupil_y = int(self.eye_center[1] + dy)

        return (pupil_x, pupil_y), (norm_x, norm_y), (target_x, target_y)

    def render_eye(self, norm_x, norm_y, face_center_x, face_center_y, pupil_pos):
        """Render the animated eye in Pygame window with enhanced debug info."""
        self.screen.fill(self.BLACK)

        # Draw the eye
        eye_color = self.GREEN if self.face_detected else self.WHITE
        pygame.draw.circle(self.screen, eye_color, self.eye_center, self.eye_radius)
        pygame.draw.circle(self.screen, self.BLACK, self.eye_center, self.eye_radius, 3)
        pygame.draw.circle(self.screen, self.BLACK, pupil_pos, self.pupil_radius)

        # Debug info
        debug_text1 = f"Norm X: {norm_x:.2f}, Norm Y: {norm_y:.2f}"
        debug_text2 = f"Face X: {face_center_x:.1f}, Y: {face_center_y:.1f}"
        debug_text3 = f"Pupil X: {pupil_pos[0]:.1f}, Y: {pupil_pos[1]:.1f}"
        debug_surface1 = self.debug_font.render(debug_text1, True, self.WHITE)
        debug_surface2 = self.debug_font.render(debug_text2, True, self.WHITE)
        debug_surface3 = self.debug_font.render(debug_text3, True, self.WHITE)
        self.screen.blit(debug_surface1, (10, 10))
        self.screen.blit(debug_surface2, (10, 30))
        self.screen.blit(debug_surface3, (10, 50))

        pygame.display.flip()
        self.clock.tick(self.fps_target)

    def process(self, stream):
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                if not flag:
                    print("Error: Could not read frame")
                    break

                # Handle Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        stream.release()
                        cv2.destroyAllWindows()
                        return

                kk = cv2.waitKey(1)
                img_rd = cv2.resize(img_rd, (320, 240))
                faces = detector(img_rd, 0)

                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                # Default values for eye tracking
                norm_x, norm_y = 0, 0
                face_center_x, face_center_y = self.feed_width // 2, self.feed_height // 2
                pupil_pos = self.eye_center

                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (self.reclassify_interval_cnt != self.reclassify_interval):
                    logging.debug("scene 1: No face cnt changes in this frame!!!")
                    self.current_frame_face_position_list = []

                    if "unknown" in self.current_frame_face_name_list:
                        self.reclassify_interval_cnt += 1

                    if self.current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])
                            img_rd = cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (255, 255, 255), 2)

                            # Eye tracking: Use facial landmarks to find the point between the eyes
                            shape = predictor(img_rd, d)
                            eye_points = []
                            for i in range(36, 48):  # Eye landmarks (36-41: right eye, 42-47: left eye)
                                x = shape.part(i).x
                                y = shape.part(i).y
                                eye_points.append((x, y))
                            if eye_points:
                                face_center_x = sum([p[0] for p in eye_points]) / len(eye_points)
                                face_center_y = sum([p[1] for p in eye_points]) / len(eye_points)

                                # Smooth the tracking point
                                self.face_detected = True
                                self.no_face_counter = 0
                                self.last_face_x = self.smoothing_alpha * face_center_x + (1 - self.smoothing_alpha) * self.last_face_x
                                self.last_face_y = self.smoothing_alpha * face_center_y + (1 - self.smoothing_alpha) * self.last_face_y

                                # Get pupil position and normalized coordinates
                                pupil_pos, (norm_x, norm_y), (face_center_x, face_center_y) = self.get_pupil_position(self.last_face_x, self.last_face_y)

                                # Draw the tracking point on the cv2 window for verification
                                cv2.circle(img_rd, (int(face_center_x), int(face_center_y)), 5, self.RED, -1)

                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()

                    for i in range(self.current_frame_face_cnt):
                        img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i],
                                             self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
                    self.draw_note(img_rd)

                else:
                    logging.debug("scene 2: Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0

                    if self.current_frame_face_cnt == 0:
                        logging.debug("  / No faces in this frame!!!")
                        self.current_frame_face_name_list = []
                        self.no_face_counter += 1
                        if self.no_face_counter > self.RESET_DELAY:
                            self.face_detected = False
                            self.last_face_x = self.feed_width // 2
                            self.last_face_y = self.feed_height // 2
                            face_center_x, face_center_y = self.last_face_x, self.last_face_y
                            pupil_pos, (norm_x, norm_y), _ = self.get_pupil_position(self.last_face_x, self.last_face_y)
                    else:
                        logging.debug("  scene 2.2  Get faces in this frame and do face recognition")
                        self.current_frame_face_name_list = []
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_name_list.append("unknown")

                        for k in range(len(faces)):
                            shape = predictor(img_rd, faces[k])
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # Eye tracking: Calculate midpoint between eyes
                            eye_points = []
                            for i in range(36, 48):
                                x = shape.part(i).x
                                y = shape.part(i).y
                                eye_points.append((x, y))
                            if eye_points:
                                face_center_x = sum([p[0] for p in eye_points]) / len(eye_points)
                                face_center_y = sum([p[1] for p in eye_points]) / len(eye_points)

                                # Smooth the tracking point
                                self.face_detected = True
                                self.no_face_counter = 0
                                self.last_face_x = self.smoothing_alpha * face_center_x + (1 - self.smoothing_alpha) * self.last_face_x
                                self.last_face_y = self.smoothing_alpha * face_center_y + (1 - self.smoothing_alpha) * self.last_face_y

                                # Get pupil position and normalized coordinates
                                pupil_pos, (norm_x, norm_y), (face_center_x, face_center_y) = self.get_pupil_position(self.last_face_x, self.last_face_y)

                                # Draw the tracking point on the cv2 window
                                cv2.circle(img_rd, (int(face_center_x), int(face_center_y)), 5, self.RED, -1)

                            self.current_frame_face_X_e_distance_list = []
                            for i in range(len(self.face_features_known_list)):
                                if str(self.face_features_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[k],
                                        self.face_features_known_list[i])
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    self.current_frame_face_X_e_distance_list.append(999999999)

                            similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                min(self.current_frame_face_X_e_distance_list))
                            if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                                nam = self.face_name_known_list[similar_person_num]
                                self.attendance(nam)
                                speak(f"नमस्ते, {self.face_name_known_list[similar_person_num]}", lang='en', speed=1.2)

                        self.draw_note(img_rd)

                # Render the eye in Pygame
                self.render_eye(norm_x, norm_y, face_center_x, face_center_y, pupil_pos)

                if kk == ord('q'):
                    break

                self.update_fps()
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", img_rd)

                logging.debug("Frame ends\n\n")

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.process(cap)
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

def main():
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()

if __name__ == '__main__':
    main()