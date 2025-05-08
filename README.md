# Smart_water_box
This repository covers the codebase for the automated image recognition pipeline created by the project proponents to extract the meter reading value from an image of a water meter using pipeline with 2 model cnn for detecing and reading.
About the Project
In many parts of the world, water utility companies still heavily rely on manual meter reading, requiring readings to be transcribed by hand, which is inefficient and leaves room for human errors. Smart metering infrastructures have been proposed as a solution, but while effective and convenient, are costly and impractical to implement on a broader scale. This work presents a solution compatible with the existing infrastructure using image recognition techniques to create an end-to-end system for automated water meter reading. This repository covers the codebase for the automated image recognition pipeline created by the project proponents to extract the meter reading value from an image of a water meter.

About the Pipeline
The pipeline follows the architechture laid out by past works on image based automatic meter reading, which is to utilize deep learning methods, and to split the task into two stages to improve the performance of the pipeline. These are the counter detection stage, which detects the location of the meter counter, and the counter recognition stage, that identifies the digits within the counter. The counter detection stage was implemented using a binary segmentation model with the U-Net architecture using a Resnet34 backbone, which was created using the Segmentation Models library. On the other hand, the counter recognition stage was implemented using an object detection model using the Faster RCNN architecture with a ResNet101 backbone created with the TensorFlow Object Detection API. Aside from these, there are extra processing stages before each of the abovementioned stages and a final post-processing stage. The final completed pipeline ca be seen in the image below.
![image](https://github.com/user-attachments/assets/25552272-e96b-476c-b9eb-775fd7a86c79)

Result
![image](https://github.com/user-attachments/assets/ba5202c2-83d3-47e5-b16c-d21189d8d8f9)
