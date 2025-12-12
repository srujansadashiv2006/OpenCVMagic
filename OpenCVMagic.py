import streamlit as st
import cv2
import numpy as np
import time
import mediapipe as mp
import pyautogui

def get_color_ranges(color_choice):
    if color_choice == 'Red':
        low1 = np.array([0, 120, 70])
        high1 = np.array([10, 255, 255])
        low2 = np.array([170, 120, 70])
        high2 = np.array([180, 255, 255])
        return low1, high1, low2, high2
    elif color_choice == 'Blue':
        return np.array([100, 150, 0]), np.array([140, 255, 255]), None, None
    elif color_choice == 'Green':
        return np.array([35, 50, 50]), np.array([85, 255, 255]), None, None
    elif color_choice == 'White':
        return np.array([0, 0, 200]), np.array([180, 55, 255]), None, None
    return None

def main():
    st.set_page_config(page_title="Harry Potter's Invisible Cloak ", layout="wide")
    st.title("Harry Potter's Invisible Cloak 🧙‍♂️")

    app_mode = st.sidebar.selectbox("Choose App Mode", ["Invisible Cloak", "Gesture Game"])

    if app_mode == "Invisible Cloak":
        st.header("Invisible Cloak")
        st.sidebar.header("Cloak Settings")
        color_choice = st.sidebar.selectbox("Choose Cloak Color:", ["Red", "Blue", "Green", "White"])
        run_cloak = st.checkbox('Start Cloak Camera')
        
        FRAME_WINDOW = st.image([])

        if run_cloak:
            lower_hsv, upper_hsv, lower_hsv2, upper_hsv2 = get_color_ranges(color_choice)
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Error: Could not open camera.")
            else:
                st.info("Capturing background... Move out of frame!")
                background = 0
                
                # Warmup and capture background
                for i in range(30):
                    ret, background = cap.read()
                
                background = np.flip(background, axis=1)
                st.success("Background captured!")

                while run_cloak:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = np.flip(frame, axis=1)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
                    if lower_hsv2 is not None:
                        mask_red = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
                        mask = mask + mask_red

                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

                    mask_inv = cv2.bitwise_not(mask)
                    res1 = cv2.bitwise_and(background, background, mask=mask)
                    res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)
                    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
                    
                    FRAME_WINDOW.image(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
                
                cap.release()

    elif app_mode == "Gesture Game":
        st.header("Gesture Controller")
        st.markdown("Use your **Index Finger** to swipe. Ensure you run this locally.")
        
        run_game = st.checkbox('Start Gesture Control')
        FRAME_WINDOW = st.image([])
        feedback_placeholder = st.empty()

        if run_game:
            # Config
            SWIPE_THRESHOLD = 50
            COOLDOWN_TIME = 0.5
            pyautogui.FAILSAFE = False

            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

            start_x, start_y = None, None
            last_action_time = 0
            
            cap = cv2.VideoCapture(0)

            while run_game:
                success, image = cap.read()
                if not success:
                    break

                image = cv2.flip(image, 1)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_image)
                
                height, width, _ = image.shape
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        # Index Finger Tip (Landmark 8)
                        index_tip = hand_landmarks.landmark[8]
                        current_x = int(index_tip.x * width)
                        current_y = int(index_tip.y * height)

                        cv2.circle(image, (current_x, current_y), 10, (0, 255, 0), -1)

                        current_time = time.time()
                        
                        if start_x is None or start_y is None:
                            start_x, start_y = current_x, current_y
                        
                        diff_x = current_x - start_x
                        diff_y = current_y - start_y

                        if (current_time - last_action_time) > COOLDOWN_TIME:
                            if abs(diff_x) > abs(diff_y):
                                if abs(diff_x) > SWIPE_THRESHOLD:
                                    if diff_x > 0:
                                        feedback_placeholder.info("➡️ Swipe Right")
                                        pyautogui.press('right')
                                    else:
                                        feedback_placeholder.info("⬅️ Swipe Left")
                                        pyautogui.press('left')
                                    
                                    last_action_time = current_time
                                    start_x, start_y = current_x, current_y
                            else:
                                if abs(diff_y) > SWIPE_THRESHOLD:
                                    if diff_y > 0:
                                        feedback_placeholder.info("⬇️ Swipe Down")
                                        pyautogui.press('down')
                                    else:
                                        feedback_placeholder.info("⬆️ Swipe Up")
                                        pyautogui.press('up')
                                    
                                    last_action_time = current_time
                                    start_x, start_y = current_x, current_y
                else:
                    start_x = None
                    start_y = None

                FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            cap.release()
            hands.close()

if __name__ == '__main__':
    main()