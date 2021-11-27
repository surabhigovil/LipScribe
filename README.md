## LipScribe: An application that converts lip movements of speakers in a silent video to text and display that using an android application. Exploiting the capabilities of 3D CNN to extract information from spatio temporal data this Deep Learning aims at creating words from a sequence of frames in a video.

# Working of the application:
1. Use the android aplication to record a video.
2. The process goes through preprocessing where Haar Cascade Classifier extract frames video and subsequently lips of a speaker from those frames.
3. This is sent to a 3D CNN model which outputs a word as the final output.

# Model Evaluation:

# Android Application:
An android operating system compatible application is developed to deploy the predictions from the model. The application requires model built with tensorflow version 1.15. ffmpeg library is used to create a sequence of frames for data preprocessing. These frames for each video are sent to the model for prediction. 

# Demo: