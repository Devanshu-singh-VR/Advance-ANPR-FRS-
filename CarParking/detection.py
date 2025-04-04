import numpy as np
from datetime import datetime
from ultralytics import YOLO
import cv2 as cv
import csv
import subprocess

# if we are using the webpage
webpage = False

if webpage:
    from model.libraries.sort.sort import *
    from model.utility import Functions
    from model.image_enhancer import ESRGAN
    from model.visualizations import CVision
else:
    from libraries.sort.sort import *
    from utility import Functions
    from image_enhancer import ESRGAN
    from visualizations import CVision

class FaceAndNumberPlateDetection:
    def __init__(self):
        if webpage:
            # loading the yolo-v8 model trained on COCO dataset
            # to capture the vehicles
            self.vehicle_detection_model = YOLO("model/yolov8n.pt")

            # loading the yolo-v8 model weights to detect number plates and faces
            # model contain 2 classes (face and vehicle number plate)
            self.face_plate_detection_model = YOLO("model/Face_Plate_weights/best.pt")
        else:
            # loading the yolo-v8 model trained on COCO dataset
            # to capture the vehicles
            self.vehicle_detection_model = YOLO("yolov8n.pt")

            # loading the yolo-v8 model weights to detect number plates and faces
            # model contain 2 classes (face and vehicle number plate)
            self.face_plate_detection_model = YOLO("Face_Plate_weights/best.pt")

        # load the ESR-GAN model to enhance the images
        # self.esr_model = ESRGAN()

        # switched for live, saved videos and for images
        self.live_video = False
        self.saved_video = False
        self.image = False

        # enhance the image in real time or at the end
        self.real_time_image_enhancing = False
        self.visualization = False

        # load a class to visualize the detected frames
        self.get_frame = CVision(self.real_time_image_enhancing)

        # vehicle class list (2-> car, 3-> motorcycle, 5-> bus, 7-> truck)
        self.vehicles_classes = [2, 3, 5, 7]

        # track the x and y shape of the images
        self.old_x = 0
        self.old_y = 0

        # get the motion tracker
        self.motion_tracker = Sort()

        # call the functions
        self.functions = Functions()

        # plate and face model class ids
        self.plate_face_class_id = {0: "plate", 1: "face", 2: "none"}

        # store the data of face and plates in results dictionary
        self.plate_results = {}
        self.face_results = {}
        self.frame_no = 0
        self.store_v_id = []

        # generate a folder with the name of current date and time
        self.date_path = str(datetime.now())
        self.date_path = self.date_path.replace(":", ".")

        # change the notepad
        file_path = 'D:\ANPR_and_FaceDetection\CarParking\saved_csv\\time.txt'  # Replace with the actual path of your text file

        # Open the file in write mode and rewrite the content
        with open(file_path, 'w') as file:
            file.write(self.date_path)

        # save face image and plate image index
        self.face_img_index = 0
        self.plate_img_index = 0

        # get the frames and socket io function
        self.socket_io = 0
        self.stop_frames = False

        if webpage:
            # paths to save the face and plate images
            self.normal_face_image_path = 'model/saved_captured_data/captured_face_images/normal_images/'
            self.enhanced_face_image_path = 'model/saved_captured_data/captured_face_images/enhanced_images/'
            self.plate_image_path = 'model/saved_captured_data/ANPR_data/number_plates_images/'
            self.enhanced_plate_image_path = 'model/saved_captured_data/ANPR_data/enhanced_number_plates_images/'
            self.vehicle_image_path = 'model/saved_captured_data/ANPR_data/vehicle_images/'
            self.csv_raw_path = 'D:\ANPR_and_FaceDetection\CarParking\saved_csv\csv_raw_data\\' + self.date_path + '.csv'
            self.csv_refine_path = 'D:\ANPR_and_FaceDetection\CarParking\saved_csv\csv_refined_data\\' + self.date_path + '.csv'
            self.csv_raw_face_path = 'model/saved_captured_data/captured_face_images/csv_face_raw_data/' + self.date_path + '.csv'
        else:
            # paths to save the face and plate images
            self.normal_face_image_path = 'model/saved_captured_data/captured_face_images/normal_images/'
            self.enhanced_face_image_path = 'model/saved_captured_data/captured_face_images/enhanced_images/'
            self.plate_image_path = 'model/saved_captured_data/ANPR_data/number_plates_images/'
            self.enhanced_plate_image_path = 'model/saved_captured_data/ANPR_data/enhanced_number_plates_images/'
            self.vehicle_image_path = 'model/saved_captured_data/ANPR_data/vehicle_images/'
            self.csv_raw_path = 'D:\ANPR_and_FaceDetection\CarParking\saved_csv\csv_raw_data\\' + self.date_path + '.csv'
            self.csv_refine_path = 'D:\ANPR_and_FaceDetection\CarParking\saved_csv\csv_refined_data\\' + self.date_path + '.csv'
            self.count_vehicle = 'D:\ANPR_and_FaceDetection\CarParking\saved_csv\count_vehicles\\' + self.date_path + '.csv'
            self.csv_raw_face_path = 'model/saved_c aptured_data/captured_face_images/csv_face_raw_data/' + self.date_path + '.csv'

    # start capturing and detecting
    def start_capturing(self, path=None):
        # make a folder to save data
        os.makedirs(self.normal_face_image_path + self.date_path)
        os.makedirs(self.enhanced_face_image_path + self.date_path)
        os.makedirs(self.plate_image_path + self.date_path)
        os.makedirs(self.enhanced_plate_image_path + self.date_path)
        os.makedirs(self.vehicle_image_path + self.date_path)

        # start capturing
        if self.live_video:
            capt = cv.VideoCapture(0)
            self.detect_video(capt)
        elif self.saved_video:
            capt = cv.VideoCapture(path)
            self.detect_video(capt)
        elif self.image:
            image = cv.imread(path)
            self.detect_image(image)

        # write all the plate data in csv format
        self.functions.plate_to_csv(self.plate_results, self.csv_raw_path)

        # refine the saved plate csv file
        self.functions.refine_plate_csv(self.csv_raw_path, self.csv_refine_path)

        # count vehicle
        self.functions.count_v(self.csv_refine_path, self.count_vehicle)

        # run a script
        # Replace 'python' with 'python3' if needed
        python_executable = 'python'

        # Path to the Python file you want to run
        python_script = 'data.py'

        # Run the Python file using subprocess
        subprocess.run([python_executable, python_script])

    # this is to detect the face and plates from a video or live video
    def detect_video(self, capture):
        while capture.isOpened():
            rets, frames = capture.read()

            # Break the loop if no frames are left
            if not rets or self.stop_frames:
                break

            # save the x and y values
            self.old_y, self.old_x, a = frames.shape

            # detect the face and plates
            out_frame = self.detection(frames, np.copy(frames))

            # show the captured data frames
            cv.imshow('video frames', out_frame)

            # install wait key
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            # increase the frame number by one for video
            self.frame_no += 1

    # this is to detect the face and plates from an image
    def detect_image(self, image):
        # detect the face and plates
        out_frame = self.detection(image, image.copy())

        # show the captured data frames
        cv.imshow('video frames', out_frame)

        # install wait key
        cv.waitKey(0)

    def detection(self, frame, copy_frame):
        # calling the face and plate detection model
        detect_face_and_plate_model = self.face_plate_detection_model(frame)[0]

        # calling the model to detect the vehicles
        detect_vehicle_model = self.vehicle_detection_model(frame)[0]

        # capture detected vehicles
        vehicle_detection = []

        # get vehicle data
        for detection in detect_vehicle_model.boxes.data.tolist():
            xvi1, yvi1, xvi2, yvi2, score, class_v_id = detection
            xvi1 = int(xvi1)
            yvi1 = int(yvi1)
            xvi2 = int(xvi2)
            yvi2 = int(yvi2)
            class_v_id = int(class_v_id)

            # make sure they are vehicles
            if class_v_id in self.vehicles_classes:
                vehicle_detection.append([xvi1, yvi1, xvi2, yvi2, score])

        # track detected vehicles
        vehicle_track_id = self.motion_tracker.update(np.asarray(vehicle_detection))

        # get face and plates data
        for plate_face_data in detect_face_and_plate_model.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = plate_face_data
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            class_id = int(class_id)

            # default vehicle values
            xv1, yv1, xv2, yv2 = 0, 0, 0, 0
            plate_number_text = ' '

            # for vehicle number plates
            if self.plate_face_class_id[class_id] == "plate":
                if self.frame_no not in self.plate_results:
                    self.plate_results[self.frame_no] = {}

                # assigning plates to the vehicle id
                xv1, yv1, xv2, yv2, v_id = self.functions.get_the_car_with_plate(plate_face_data, vehicle_track_id)
                xv1 = int(xv1)
                yv1 = int(yv1)
                xv2 = int(xv2)
                yv2 = int(yv2)

                # get the plates
                license_plate = frame[y1:y2, x1:x2, :]

                # scale the image
                scale_image = cv.resize(license_plate, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

                # convert plate to gray scale
                grey_license_plate = cv.cvtColor(scale_image, cv.COLOR_BGR2GRAY)

                # add inverse threshold (black -> white , white -> black)
                _, threshold_license_plate = cv.threshold(grey_license_plate,
                                                          160, 255, cv.THRESH_BINARY)

                """ SAVE CAR IMAGE """
                if v_id != -1:
                    # crop the car image from frame
                    car_image = frame[yv1:yv2, xv1:xv2, :]

                    # path to save car image
                    save_img_path = self.vehicle_image_path + self.date_path \
                                    + '/' + str(self.plate_img_index) + ".jpg"

                    # save the car image
                    cv.imwrite(save_img_path, car_image)

                # increase save plate image index
                self.plate_img_index += 1

                ''' STORE THE CAR PLATE DATA '''
                # read license plate
                plate_number_text, plate_text_score = self.functions.read_number_plate_text(grey_license_plate)

                if plate_number_text is not None:
                    self.plate_results[self.frame_no][v_id] = {'car': {'bbox': [xv1, yv1, xv2, yv2]},
                                                               'license plate': {'bbox': [x1, y1, x2, y2],
                                                                                 'text': plate_number_text,
                                                                                 'bbox score': score,
                                                                                 'text score': plate_text_score,
                                                                                 'time stamp': self.date_path}}

            elif self.plate_face_class_id[class_id] == "face":
                # Initialize the inner dictionary only once per frame_no
                if self.frame_no not in self.face_results:
                    self.face_results[self.frame_no] = {}

                self.face_results[self.frame_no][self.face_img_index] = {}

                # assigning plates to the vehicle id
                xv1, yv1, xv2, yv2, v_id = self.functions.get_the_car_with_plate(plate_face_data, vehicle_track_id)
                xv1 = int(xv1)
                yv1 = int(yv1)
                xv2 = int(xv2)
                yv2 = int(yv2)

                # get the face image
                face_image = frame[y1:y2, x1:x2, :]

                ''' STORE THE FACE CAR DATA '''
                self.face_results[self.frame_no][self.face_img_index][v_id] = {'car': {'bbox': [xv1, yv1, xv2, yv2]},
                                                                               'face': {'bbox': [x1, y1, x2, y2],
                                                                                        'bbox score': score,
                                                                                        'time stamp': self.date_path}}

                # increase the save face image index
                self.face_img_index += 1

            # get the detected box frames and text
            copy_frame = self.get_frame.show_image(copy_frame, [x1, y1, x2, y2],
                                                   [xv1, yv1, xv2, yv2], class_id,
                                                   self.old_x, self.old_y, plate_number_text)

        return copy_frame


# to call the class
if __name__ == '__main__':
    model = FaceAndNumberPlateDetection()
    model.saved_video = True
    model.image = False
    model.live_video = False
    model.real_time_image_enhancing = False
    model.visualization = False

    model.start_capturing("entry1.mp4")
    print(model.plate_results)
    print(model.face_results)


