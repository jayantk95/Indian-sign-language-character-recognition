# imports
import numpy as np
import cv2
import pickle
import ImagePreProcessing as ipu
import tensorflow

CAPTURE_FLAG = False

class_labels = ipu.get_labels()


def recognise():
    global CAPTURE_FLAG
    # Load the model
    model = tensorflow.keras.models.load_model('model_final.h5')

    gestures = ipu.get_all_gestures()
   # print(gestures)
    cv2.imwrite("all_gestures.jpg", gestures)
    camera = cv2.VideoCapture(0)
    print(
        'Now camera window will be open, then \n1) Place your hand gesture in ROI (rectangle) \n2) Press esc key to exit.')
    count = 0
    while (True):
        (t, frame) = camera.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, ipu.START, ipu.END, (0, 255, 0), 2)
        cv2.imshow("All_gestures", gestures)
        pressedKey = cv2.waitKey(1)
        if pressedKey == 27:
            break
        elif pressedKey == ord('q'):
            if (CAPTURE_FLAG):
                CAPTURE_FLAG = False
            else:
                CAPTURE_FLAG = True
        if (CAPTURE_FLAG):
            # Region of Interest
            roi = frame[ipu.START[1] + 5:ipu.END[1], ipu.START[0] + 5:ipu.END[0]]
            if roi is not None:
                roi = cv2.resize(roi, (ipu.IMG_SIZE, ipu.IMG_SIZE))
                img = ipu.get_canny_edge(roi)[0]
                cv2.imshow("Edges ", img)
                print(img)
            img = img.reshape((1, 128, 128, 1))
            pred = model.predict(img)


            score = tensorflow.nn.softmax(pred[0])
            label = class_labels[np.argmax(score)]
            rectangle_bgr = (0, 0, 0)
            (text_width, text_height) = cv2.getTextSize('Predicted text:      ', 1, fontScale=1.5, thickness=2)[0]
            # set the text start position
            text_offset_x = 50
            text_offset_y = 20
            # make the coords of the box with a small padding of two pixels
            box_coords = (
            (text_offset_x, text_offset_y), (text_offset_x + text_width + 40, text_offset_y + text_height + 50))
            cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
            frame = cv2.putText(frame, 'Predicted text: ', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                    2, cv2.LINE_AA)
            frame = cv2.putText(frame, label, (300, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                                    cv2.LINE_AA)

        cv2.imshow("Video", frame)
    camera.release()
    cv2.destroyAllWindows()

recognise()





