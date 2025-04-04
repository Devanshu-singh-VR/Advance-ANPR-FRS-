import easyocr
import csv
import pandas as pd

class Functions:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.good = None

    def get_the_car_with_face(self, face_plate_data, vehicle_track_id):
        # get the score and bbox of face
        x1, y1, x2, y2, score, class_id = face_plate_data

        for idx in range(len(vehicle_track_id)):
            xv1, yv1, xv2, yv2, v_id = vehicle_track_id[idx]

            # the range of plate_bbox should be in bbox of particular car
            if self.face_in_range([x1, y1, x2, y2], [xv1, yv1, xv2, yv2]):
                return vehicle_track_id[idx]

        return -1, -1, -1, -1, -1

    # To check weather plate bbox comes in which vehicle bbox
    def face_in_range(self, f_bbox, v_bbox):
        # to get the middle y coordinate of the vehicle
        vy_half = (v_bbox[3] + v_bbox[1]) / 2

        if f_bbox[0] > v_bbox[0] and f_bbox[1] > v_bbox[1] and \
            f_bbox[2] < v_bbox[2] and f_bbox[3] < v_bbox[3] and \
            f_bbox[1] <= vy_half and f_bbox[3] <= vy_half: \
            return True
        else:
            return False

    def get_the_car_with_plate(self, face_plate_data, vehicle_track_id):
        # get the score and bbox of plate
        x1, y1, x2, y2, score, class_id = face_plate_data

        for idx in range(len(vehicle_track_id)):
            xv1, yv1, xv2, yv2, v_id = vehicle_track_id[idx]

            # the range of plate_bbox should be in bbox of particular car
            if self.plate_in_range([x1, y1, x2, y2], [xv1, yv1, xv2, yv2]):
                return vehicle_track_id[idx]

        return -1, -1, -1, -1, -1

    # To check weather plate bbox comes in which vehicle bbox
    def plate_in_range(self, plt_bbox, v_bbox):
        if plt_bbox[0] > v_bbox[0] and plt_bbox[1] > v_bbox[1] and \
            plt_bbox[2] < v_bbox[2] and plt_bbox[3] < v_bbox[3]: \
            return True
        else:
            return False

    # to read the text on plates
    def read_number_plate_text(self, plate_cropped_image):
        # call the easyocr function
        detections = self.reader.readtext(plate_cropped_image)

        # default values
        text = "None"
        score = -1

        # extract the data
        for data in detections:
            bbox, text, score = data

        # convert text to upper case
        text = str(text).upper()

        return text, score

    # convert plate dictionary data to csv format for better view
    def plate_to_csv(self, inp, path):
        with open(path, mode='w', newline='') as csv_file:
            fieldnames = ['frame_number', 'vehicle_id', 'vehicle_bbox_coordinates', 'plate_bbox_coordinates',
                          'plate_bbox_score', 'plate_number', 'plate_number_score', 'time_stamp']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for frame_nmr, frame_data in inp.items():
                for vehicle_id, vehicle_data in frame_data.items():
                    car_data = vehicle_data.get('car', {})
                    license_plate_data = vehicle_data.get('license plate', {})

                    row = {
                        'frame_number': frame_nmr,
                        'vehicle_id': vehicle_id,
                        'vehicle_bbox_coordinates': car_data.get('bbox', []),
                        'plate_bbox_coordinates': license_plate_data.get('bbox', []),
                        'plate_bbox_score': license_plate_data.get('bbox score', 0),
                        'plate_number': license_plate_data.get('text', ''),
                        'plate_number_score': license_plate_data.get('text score', 0),
                        'time_stamp': license_plate_data.get('time stamp', ''),
                    }

                    writer.writerow(row)

    # refine the score of plate data in csv
    def refine_plate_csv(self, input_path, output_path):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(input_path)

        # Group the DataFrame by 'vehicle_id' and find the row with the highest 'plate_number_score' in each group
        highest_scores_df = df.loc[df.groupby('vehicle_id')['plate_number_score'].idxmax()]

        # Save the DataFrame with the highest scores to a new CSV file
        highest_scores_df.to_csv(output_path, index=False)

    # store the face dictionary data to csv format
    def face_to_csv(self, inp, path):
        with open(path, mode='w', newline='') as csv_file:
            fieldnames = ['frame_number', 'face_img_idx', 'vehicle_id', 'vehicle_bbox_coordinates',
                          'face_bbox_coordinates', 'face_bbox_score', 'time_stamp']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for frame_nmr, frame_data in inp.items():
                for face_img_idx, face_data in frame_data.items():
                    for vehicle_id, vehicle_data in face_data.items():
                        car_data = vehicle_data.get('car', {})
                        face_data = vehicle_data.get('face', {})

                        row = {
                            'frame_number': frame_nmr,
                            'face_img_idx': face_img_idx,
                            'vehicle_id': vehicle_id,
                            'vehicle_bbox_coordinates': car_data.get('bbox', []),
                            'face_bbox_coordinates': face_data.get('bbox', []),
                            'face_bbox_score': face_data.get('bbox score', 0),
                            'time_stamp': face_data.get('time stamp', '')
                        }

                        writer.writerow(row)

    def count_v(self, path_in, path_out):
        data = pd.read_csv(path_in)

        datas = {"number of cars": [], "plate number": []}

        no_cars = len(data)

        texts = data.iloc[:, 5]
        for i in range(no_cars):
            datas["number of cars"].append(i+1)
            datas["plate number"].append(str(texts[i]))

        # save the file
        clf = pd.DataFrame(datas)
        clf.to_csv(path_out, index=False)

if __name__ == '__main__':
    print("utility directory")