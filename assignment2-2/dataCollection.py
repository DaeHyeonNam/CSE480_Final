import cv2
import mediapipe as mp
import keyboard
import csv
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
while cap.isOpened():
  success, image = cap.read()
  if not success:
    break

  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  results = hands.process(image)

  # Draw the hand annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if results.multi_hand_landmarks:
    if( keyboard.is_pressed('a')):
        with open('data.csv', mode='a') as data_file:
            data_writer =  csv.writer(data_file, delimiter=',')
            landmark_list = []
            for idx, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
                landmark_list.append(landmark.x)
                landmark_list.append(landmark.y)
            landmark_list.append("1")
            data_writer.writerow(landmark_list)
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
          image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
  cv2.imshow('MediaPipe Hands', image)
  if cv2.waitKey(5) & 0xFF == 27:
    break
hands.close()
cap.release()
