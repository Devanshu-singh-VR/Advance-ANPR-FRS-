import os
import re

def convert_txt_to_prototxt(input_file, output_file):
    # Read the content of the .txt file
    with open(input_file, 'r') as txt_file:
        txt_content = txt_file.read()

    # Convert the content to Protobuf format
    protobuf_content = convert_to_protobuf(txt_content)

    # Save the Protobuf content to the .prototxt file
    with open(output_file, 'w') as prototxt_file:
        prototxt_file.write(protobuf_content)

def convert_to_protobuf(txt_content):
    # Implement your logic to convert txt content to Protobuf format
    # This can involve regular expressions or simple string manipulation
    # Replace this example with your actual conversion logic
    protobuf_content = txt_content.replace('txt', 'protobuf')
    return protobuf_content

# Specify input and output file paths
input_txt_file = "D:\ANPR_and_FaceDetection\\age&gender\\GenderNet.txt"
output_prototxt_file = 'D:\ANPR_and_FaceDetection\\age&gender\\gen.prototxt'

# Convert .txt to .prototxt
convert_txt_to_prototxt(input_txt_file, output_prototxt_file)
