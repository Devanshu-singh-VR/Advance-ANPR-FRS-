from detection import FaceAndNumberPlateDetection

# run the detection.py
model = FaceAndNumberPlateDetection()
model.saved_video = False
model.image = True
model.live_video = False
model.real_time_image_enhancing = False
model.visualization = False

model.start_capturing('arab.jpg')
print(model.plate_results)
print(model.face_results)



