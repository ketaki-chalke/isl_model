import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load trained LSTM model
model = load_model("P1_model.h5")

# Define gesture classes (updated)
actions = np.array(['Hello', 'Good', 'Morning', 'Help', 'House', 'Thankyou', 'Nice', 'Welcome', 'Yes', 'No'])

# Generate random colors dynamically for each action
colors = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in range(len(actions))]

# Initialize Mediapipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to process frame with Mediapipe
def mediapipe_detection(image, holistic_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to extract keypoints safely
def extract_keypoints(results):
    try:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
        return np.concatenate([pose, face, lh, rh])
    except Exception as e:
        print(f"Error in extracting keypoints: {e}")
        return np.zeros(1662)

# Function to visualize probabilities
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, f"{actions[num]}: {prob:.2f}", (5, 85 + num * 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Open webcam
cap = cv2.VideoCapture(0)

sequence = []  # Stores last 30 frames (updated)
sentence = ""  # Stores recognized word

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Process frame with Mediapipe
        image, results = mediapipe_detection(frame, holistic)

        # Draw keypoints
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract keypoints and store in sequence
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep last 30 frames (updated)

        # Prediction when sequence is ready
        if len(sequence) == 30:
            model_input = np.expand_dims(sequence, axis=0)
            try:
                res = model.predict(model_input)[0]  # Get prediction probabilities
                predicted_word = actions[np.argmax(res)]  # Get predicted word
                sentence = predicted_word if max(res) > 0.5 else "Unknown"  # Confidence threshold

                print(f"Detected Gesture: {sentence}")

                # Visualize probability bar
                image = prob_viz(res, actions, image, colors)

            except Exception as e:
                print(f"Model prediction error: {e}")

        # Display recognized word
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, sentence, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show output
        cv2.imshow('Real-time Gesture Recognition', image)

        # Press 'q' to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
