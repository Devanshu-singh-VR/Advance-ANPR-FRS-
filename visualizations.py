import cv2 as cv
import numpy as np

class CVision:
    def __init__(self, real_time_enhancer):
        # when real time enhancer is True then switch is off to show the detected face on screen
        self.switch = not real_time_enhancer

        # plate and face model class ids
        self.plate_face_class_id = {0: "plate", 1: "face", 2: "none"}

    def manage_frame(self, copy_frame, plt_f_coordinates, vehicle_coordinates, old_x, old_y):
        # get frame
        new_frame = copy_frame

        # get coordinates
        x1, y1, x2, y2 = plt_f_coordinates
        xv1, yv1, xv2, yv2 = vehicle_coordinates

        # get the shapes
        y_shape, x_shape, channels = new_frame.shape

        # manage the frame size according to the system resolution
        if y_shape >= 700:
            # for face and plate coordinates
            y1 = (y1 / old_y) * 700
            y2 = (y2 / old_y) * 700

            # for vehicle coordinates
            yv1 = (yv1 / old_y) * 700
            yv2 = (yv2 / old_y) * 700

            # resize the frame
            new_frame = cv.resize(new_frame, (x_shape, 700))
            y_shape = 700

        if x_shape >= 1200:
            # for face and plate coordinates
            x1 = (x1 / old_x) * 1200
            x2 = (x2 / old_x) * 1200

            # for vehicle coordinates
            xv1 = (xv1 / old_x) * 1200
            xv2 = (xv2 / old_x) * 1200

            # resize the frame
            new_frame = cv.resize(new_frame, (1200, y_shape))
            x_shape = 1200

        plt_f_coordinates = [int(x1), int(y1), int(x2), int(y2)]
        vehicle_coordinates = [int(xv1), int(yv1), int(xv2), int(yv2)]

        return new_frame, plt_f_coordinates, vehicle_coordinates

    def show_image(self, copy_frame, plt_f_coordinates,
                   vehicle_coordinates, class_id, old_x,
                   old_y, plate_num_text):
        # manage the frames
        new_frame, plt_f_coordinates,\
         vehicle_coordinates = self.manage_frame(copy_frame, plt_f_coordinates,
                                                 vehicle_coordinates, old_x, old_y)

        # extract the frames
        x1, y1, x2, y2 = plt_f_coordinates
        xv1, yv1, xv2, yv2 = vehicle_coordinates

        # get the text image
        text_img = self.text_to_image(plate_num_text)

        # visualization frames
        if self.plate_face_class_id[class_id] == "face":
            new_frame = cv.rectangle(new_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            new_frame = cv.rectangle(new_frame, (xv1, yv1), (xv2, yv2), (0, 255, 0), 2)
        elif self.plate_face_class_id[class_id] == "plate":
            new_frame = cv.rectangle(new_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            new_frame = cv.rectangle(new_frame, (xv1, yv1), (xv2, yv2), (0, 255, 0), 2)

            try:
                # add the plate text image to original image
                paste_x = int(((x1 + x2)/2) - (text_img.shape[1]/2))
                paste_y = y1 - text_img.shape[0]  # Pasting above the bounding box

                # Perform the paste operation
                new_frame[paste_y: (paste_y + text_img.shape[0]),
                paste_x: (paste_x + text_img.shape[1])] = text_img
            except:
                print("the plate text image is above the image")

        return new_frame

    def text_to_image(self, text):
        # Create a blank black image of size 200x100 (width x height)
        image_width, image_height = 250, 40
        blank_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        # Choose the font and its properties
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2

        # Get the size of the text
        text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]

        # Calculate the position to center the text on the image
        text_x = (image_width - text_size[0]) // 2
        text_y = (image_height + text_size[1]) // 2

        # Draw the text on the image in white color
        cv.putText(blank_image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

        return blank_image

if __name__ == '__main__':
    img = np.zeros((900, 1600, 3))
    model = CVision(True)
    frame = model.show_image(img, [0, 0, 800, 450], [800, 450, 1600, 900], 1, 1600, 900, "baby")
    text = 'rj14cg9790'
    text = text.upper()
    imgs = model.text_to_image(text)
    cv.imshow('jsj', imgs)
    # cv.imshow("img", frame)
    # cv.imshow('new', img)
    cv.waitKey(0)