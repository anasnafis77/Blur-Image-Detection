# Blur-Image-Detection

Blur image detection is essential for computer vision system to avoid misinterpretation of the image. I build blur detection script on python using focus measure and Convolutional Neural Network. This work is my project in Jakarta Smart City as Data Science Trainee. The sample dataset originated from [here](https://www.kaggle.com/kwentar/blur-dataset)

## Usage
1. Run this on command prompt:
  `python detectBlur.py`
3. Choose your image from dialog appeared.
4. After a few seconds, the blur detection result will be printed in command prompt and json file.
5. To change the blur detection, edit method variable in detectBlur.py in the line below if __name__ == '__main__':. Currently, there are two method, which are 'cnn' and 'tenengrad'. (Next update will require specifying the method as an argument in command prompt)
