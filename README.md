## LipScribe: An application that converts lip movements of speakers in a silent video to text and display that using an android application. Exploiting the capabilities of 3D CNN to extract information from spatio temporal data this Deep Learning aims at creating words from a sequence of frames in a video.

# Working of the application:
1. Use the android aplication to record a video.
2. The process goes through preprocessing where Haar Cascade Classifier extract frames video and subsequently lips of a speaker from those frames.
3. This is sent to a 3D CNN model which outputs a word as the final output.

# Model Evaluation:

![image](https://user-images.githubusercontent.com/10840984/143726926-f397b1ab-b195-4f4a-b50c-6edc0cf80a54.png)


# Android Application:
An android operating system compatible application is developed to deploy the predictions from the model. The application requires model built with tensorflow version 1.15. ffmpeg library is used to create a sequence of frames for data preprocessing. These frames for each video are sent to the model for prediction. 

# Demo:
Deetcting a word on app:
![Screenshot_20211123_143827](https://user-images.githubusercontent.com/10840984/143726831-a7cdd624-aadd-458f-a1a0-2990a318baf1.png)
![Screenshot_20211123_143929](https://user-images.githubusercontent.com/10840984/143726841-f88224b0-c95f-4cfd-8051-e5c55b899091.png)
<img width="501" alt="Screenshot 2021-11-23 at 3 22 44 PM" src="https://user-images.githubusercontent.com/10840984/143726901-604f68d9-ccca-474f-935f-caefc8b7b2d6.png">

No Faace Detection on App:
![image](https://user-images.githubusercontent.com/10840984/143727120-8ec95f3c-b26c-40c0-a7f9-2c1544ea05bf.png)
<img width="496" alt="Screenshot 2021-11-23 at 3 23 53 PM" src="https://user-images.githubusercontent.com/10840984/143726904-89e8b691-b5a8-42fe-85b5-4af4f2e5ff24.png">
