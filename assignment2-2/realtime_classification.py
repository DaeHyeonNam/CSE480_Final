import cv2
import mediapipe as mp
import keyboard
from FCN import FullyConnectedNet
import torch

weight_path = "/Users/daehyeon/Desktop/수업/컴퓨터비전/final/weights/"


font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

pretrained_net = FullyConnectedNet()
pretrained_net.load_state_dict(torch.load(weight_path+"weight_100_model.pth"))

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
    
    # do prediction
    landmark_list = []
    for idx, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
      landmark_list.append(landmark.x)
      landmark_list.append(landmark.y)
    # Calculate avg value of x and y of one set of landmarks
    x_sum = 0
    y_sum = 0
    for i in range(21):
        x_sum += float(landmark_list[i])
        y_sum += float(landmark_list[2*i+1])
    x_avg = x_sum/21
    y_avg = y_sum/21
        
    # move to 0,0 based on the average value
    processed_data = []
    for i in range(21):
        processed_data.append(float(landmark_list[i]) - x_avg)
        processed_data.append(float(landmark_list[2*i+1]) - y_avg)
        
    pred = pretrained_net(torch.FloatTensor(processed_data))
    pred = int(pred.argmax())
    if(pred == 0):
        pred_text = "paper"
    elif(pred == 1):
        pred_text = "rock"
    else:
        pred_text = "scissors"
    
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
          image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
  else:
    pred_text = ""
  cv2.putText(image,pred_text, bottomLeftCornerOfText, font, fontScale,fontColor, lineType)
  cv2.imshow('MediaPipe Hands', image)
  if cv2.waitKey(5) & 0xFF == 27:
    break
hands.close()
cap.release()
