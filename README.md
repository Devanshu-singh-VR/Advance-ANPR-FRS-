# YOLOv8-Based-Advanced-ANPR-&-FRS

▪ Introducing our advanced Deep Learning model, designed to provide a comprehensive technological solution for Automatic Number Plate
Recognition (ANPR) and Facial Recognition, delivering precise identification and tracking of vehicles and individuals from CCTV feeds with unparalleled
accuracy.

![gifff](https://github.com/user-attachments/assets/95c7675f-9e6d-47d7-b305-08965c254caf)

▪ For Automatic Number Plate Recognition, we will use YOLOv8 for the number plate detection, further the detected number plate image will be
passed through the ESRGAN to increase the resolution of the image, and a grid search to detect the characters and numbers on the plate. We will train
a character and digit recognition RCNN model which can detect any character with different languages and numbers.

# Accuracy and Loss Plots 
![Screenshot (74)](https://github.com/user-attachments/assets/5a99313d-3981-4541-a134-7e79202f4197)

▪ To ensure accurate recognition of number plates in typical non-standard ways, our solution will be trained on a diverse dataset containing a variety of
vehicles, including cars, two-wheelers, mini trucks, auto-rickshaws, and lorries, with different layouts of number plates. We understand that different
types of vehicles have number plates of various shapes and sizes, and we aim to cater to this by detecting number plates for different conditions such
as different font sizes, font styles, handwritten characters, and background colors for all Indian number plates. The dataset will be vast and
comprehensive, catering to challenges such as the use of multiple types of license plates, hanging plates at different heights, different colors and sizes
of plates, and the use of non-standard fonts and templates. With this diverse dataset, our solution will be able to accurately recognize number plates
that are difficult to recognize by existing ANPR systems, providing unparalleled accuracy for identification and tracking of vehicles and individuals.

▪ In addition to recognizing number plates, our solution will take face recognition to a further level by detecting the age and gender of the recognized
face. This added feature will provide novelty to our solution, making it a valuable tool for crime CCTV footage analysis. For instance, in the event of a
robbery, the system can accurately identify the gender and age of the suspects and provide this information to law enforcement agencies to aid in
their investigations. The age detection feature can also be used to prevent age-restricted activities, such as the sale of alcohol to minors. With these
capabilities, our solution will not only provide accurate identification and tracking of vehicles but will also offer advanced face recognition capabilities,
making it a valuable tool for law enforcement agencies, businesses, and public safety organizations

# Model Workflow

![Screenshot (72)](https://github.com/user-attachments/assets/06c39ffb-9d58-4a00-a93a-b70d2c8d2fa1)





