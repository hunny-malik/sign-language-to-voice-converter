import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function to speak text in Gujarati
def speak_text_gujarati(text):
    # Set the voice to Gujarati
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'gu' in voice.languages:
            engine.setProperty('voice', voice.id)
            break
    else:
        print("Gujarati voice not found. Using default voice.")
    engine.say(text)
    engine.runAndWait()

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

sen = ''
last_detected = None  # Variable to store the last detected character
hand_detected = False  # Flag to indicate if a hand is currently detected
prediction_buffer = []  # Buffer to smooth out predictions
buffer_length = 5  # Length of the prediction buffer
frame_skip = 3  # Skip every n-th frame to reduce workload
frame_count = 0
cooldown = 20  # Cooldown period to prevent duplicate predictions
cooldown_timer = 0

# Variables for blinking effect
blink_timer = 0
blink_duration = 5  # Duration of blinking effect in frames
box_color = (0, 0, 0)  # Default box color (black)

# Initialize video capture with the correct device index
cap = cv2.VideoCapture(0)

# Check if the video capture opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Initialize Mediapipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, 
                        min_detection_confidence=0.6,  # Increased confidence level
                        min_tracking_confidence=0.5)   # Tracking confidence to ensure better results

labels_dict = {0: 'A', 1: 'B', 2: '-', 3: '_', 4: '/', 5: ':'}

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip this frame to reduce load

    # Ensure the frame is not None before processing
    if frame is None:
        print("Warning: Received an empty frame.")
        continue

    # Get the dimensions of the frame
    H, W, _ = frame.shape

    # Convert the frame to RGB for Mediapipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe to detect hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) > 1:
            # Display a warning if more than one hand is detected
            cv2.putText(frame, "Warning: Multiple hands detected!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            hand_detected = True  # Set the flag if a hand is detected

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            # Prepare the data for prediction
            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Make predictions using the loaded model
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Add the prediction to the buffer
            prediction_buffer.append(predicted_character)
            if len(prediction_buffer) > buffer_length:
                prediction_buffer.pop(0)  # Maintain buffer length

            # Check if the majority of predictions in the buffer are the same
            if cooldown_timer == 0 and prediction_buffer.count(predicted_character) > buffer_length // 2:
                if predicted_character == '-' and sen != '':
                    sen = sen[:-1]
                    # Start blinking effect
                    blink_timer = blink_duration
                    box_color = (0, 255, 0)  # Green color
                elif predicted_character == '_':
                    sen += ' '
                    # Start blinking effect
                    blink_timer = blink_duration
                    box_color = (0, 255, 0)  # Green color
                elif predicted_character == '/':
                    continue
                else:
                    sen += predicted_character
                    # Start blinking effect
                    blink_timer = blink_duration
                    box_color = (0, 255, 0)  # Green color
                cooldown_timer = cooldown  # Start cooldown timer
                last_detected = predicted_character

            # If the character is 'B', speak the sentence in Gujarati
            if predicted_character == ':':
                speak_text_gujarati(sen)
                sen = ""  # Clear the sentence after speaking

            # Draw a rectangle and put text on the frame
            if blink_timer > 0:
                # Draw the rectangle with the blinking color
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)
                blink_timer -= 1
                if blink_timer == 0:
                    box_color = (0, 0, 0)  # Reset box color to black after blinking
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    else:
        # If no hand is detected, reset the hand_detected flag and last_detected character
        if hand_detected:
            hand_detected = False
            last_detected = None
            time.sleep(1)  # Small delay to avoid detecting the same gesture immediately

    # Display the sentence above the rectangle
    cv2.putText(frame, sen, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)

    # Handle the cooldown timer
    if cooldown_timer > 0:
        cooldown_timer -= 1

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
