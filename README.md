## LipScribe: An application that converts lip movements of speakers in a silent video to text and display that using an android application. Exploiting the capabilities of 3D CNN to extract information from spatio temporal data this Deep Learning aims at creating words from a sequence of frames in a video.
# Android application installation and using:
1. Install the android application using the APK from https://drive.google.com/drive/folders/10pGHK0VYddb7Kn0rjqMDCVR4Nh-bR_U3?usp=sharing
2. On launching the application on android phone, it checks for the camera and requests the permission for recording videos and capturing pictures. 
3. Choose to allow the application to record videos and capture pictures. 
4. Click on allow for the application on request for accessing the files and media.
5. Click on 'start camera' on start page to start recording the video.
6. The recorded video is processed from external storage and given to model for prediction this happens in background and loading screen is displayed on screen.
7. The prediction of the word speaker utterted is displayed on the screen.

# Working of the application:
1. Use the android aplication to record a video.
2. The process goes through preprocessing where Haar Cascade Classifier extract frames video and subsequently lips of a speaker from those frames.
3. This is sent to a 3D CNN model which outputs a word as the final output.

# Model Evaluation:

![image](https://user-images.githubusercontent.com/10840984/143726926-f397b1ab-b195-4f4a-b50c-6edc0cf80a54.png)


# Android Application:
An android operating system compatible application is developed to deploy the predictions from the model. The application requires model built with tensorflow version 1.15. ffmpeg library is used to extract frames for data preprocessing from a video. The mouth region is extracted and converted into embeddings and passed as input to the model . 

# Demo:
Deetcting a word on app:
![Screenshot_20211123_143827](https://user-images.githubusercontent.com/10840984/143726831-a7cdd624-aadd-458f-a1a0-2990a318baf1.png)
![Screenshot_20211123_143929](https://user-images.githubusercontent.com/10840984/143726841-f88224b0-c95f-4cfd-8051-e5c55b899091.png)
<img width="501" alt="Screenshot 2021-11-23 at 3 22 44 PM" src="https://user-images.githubusercontent.com/10840984/143726901-604f68d9-ccca-474f-935f-caefc8b7b2d6.png">

No Faace Detection on App:
![image](https://user-images.githubusercontent.com/10840984/143727120-8ec95f3c-b26c-40c0-a7f9-2c1544ea05bf.png)
<img width="496" alt="Screenshot 2021-11-23 at 3 23 53 PM" src="https://user-images.githubusercontent.com/10840984/143726904-89e8b691-b5a8-42fe-85b5-4af4f2e5ff24.png">
