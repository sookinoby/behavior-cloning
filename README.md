# H1 Objective:
To train a deep neural network that can mimic the driving of a Human being (only the steering angle).

# H1 Data Collection
We used the beta version of the simulator to collect my training data. The beta version only records the center camera image with the current steering angle. The steering angle (-25 to 25 degree) is linearly mapped onto (-1 to 1). The path to the image, steering angle, throttle, brakes were stored in a CSV and we were only concerned about the image and steering angle.

 We drove around for 10 laps around the first track. We also collected data for recovery. For collecting recovery data we drove the car to the left/right edge of the road (without recording), then we drove back to the center while recording. We collected around 30 K samples for training purpose.

I also downloaded training data provided by Udacity.  Since Udacity data had all three camera images (left, center and right) I had to adjust to the steering angle of the left and right image. I subtracted 0.25 from the steering angle for the left image and added 0.25 to the right image. I choose 0.25 as the angle thanks to [Vivek Yadav]( https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee#.vjrzdttix). Below is the code snippet. 

```python
def read_csv_and_parse_three_images(image_steer,file_name,folder,path_split=True):
    counter = 0
    with open(folder+"/"+file_name) as f:
        reader = csv.reader(f)
        for row in reader:
            image_name = folder+"/"+(row[0])
            angle = float(row[3])
            image_steer[image_name] = angle
            image_name = folder + "/" + (row[1]).strip()
            image_steer[image_name] = angle + .25

            image_name = folder + "/" + (row[2]).strip()
            image_steer[image_name] = angle - .25
            counter = counter + 1
        print(counter)
    return image_steer
```
![Alt text](images/left-right-cam?raw=true "histogram")

In total there were around 45 K examples in my training set. I also recorded 2 laps for validation and two laps for testing purpose. 



### H3 The Zero bias problem
While driving the simulator, we noticed that the steering angle from -0.25 to 0.25 occurs too often. Below the histogram of steering with a bin width of 0.1. This will cause the car to favor angles from -025 to 0.25 and prevent the car from taking steep turns.  Below is the histogram
![Histogram](images/orginal_hist_all_data.png?raw=true "histogram")
 
To fix this, we need to select training data that favors higher steering angle over lower steering. We were not able to perform this operation for the whole data set at one go due to memory limitation. We used Python generator and did the following to set one sample from the training batch .
```python
def select_random(X_train,y_train,bias):
    m = np.random.randint(0, len(y_train))
    n = np.random.randint(0, 80)
    image = X_train[m]
    steering = y_train[m];
    while ((steering > -bias and steering < bias) and n > 40):
        m = np.random.randint(0, len(y_train))
        image = X_train[m]
        steering = y_train[m]
    return image, steering
```

```python
# a python generator that select images at random with some random augmentation
def generate_train_batch(X_train, y_train, batch_size=32):
    batch_images = np.zeros((batch_size, height, width, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        bias = 0
        for i_batch in range(batch_size):
            bias = i_batch / (i_batch + 10)
            x, y = generate_training_example(X_train, y_train,bias)
            image = make_roi(x)
            batch_images[i_batch] = image
            batch_steering[i_batch] = y
        yield batch_images, batch_steering
```
The output of the above function will create an evenly distrusted set of steering angles
 ![Histogram](images/norm-hist.png?raw=true "Processed histogram")

### H3 Image preprocessing:
Flipping images:
Since there are more right-turns in the simulated track, We randomly flipped image (horizontally) along with steering. However, this technique didn’t give any better model. Below is the code that flips image and the result



```python
def random_flip(image, steering):
    n = np.random.randint(0, 2)
    if n == 0 and steering < 0:
        image, steering = cv2.flip(image, 1), -steering
    return image, steering
```

![flipped images](images/flip.png?raw=true "flipped images")

### H3 Translated images(shifting images):
This was the trump card for making my model work. We still don’t understand why this technique is so effective. This function translates the car being at a different position, the steering angle is offset correspondingly. Thanks to [Vivek Yadav]( https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee#.vjrzdttix). Below is the code for translation and the result. 
```python
def random_trans(image, steer, trans_range):
    rows,cols,_ = image.shape;
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

    return image_tr, steer_ang
```

![Translated images](images/translated.png?raw=true "Translated images")


# H1 The Neural Network Architecture:
The neural network architecture we used was similar to [Nvidia’s model] (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). We added a dropout layer to meet the requirement of this project. One thing we noticed during our trial is that any neural network which is deep enough( 10 layers, with 4 convolution layers at least) is good enough.  We found that the most important factor is the training data.  Below is my model architecture:
![Model](images/model.png?raw=true "Neural network")
The input to the model is a three channel color image in the RGB format of 64 x 64 input size. The initial image is cropped to remove horizon and resized to 64X64 pixels.
![Cropped image](images/cropped-final.png?raw=true "Cropped image")

 I have used a Lamba layer to scale the input pixel range to (-1 to 1) range. This will help the optimizer to converge faster.  The convolution layers are of varying depth and filter size (3X3) and (5x5) were used. We have used stride length of 2 for all convolution layer expect one. I have used ELU as my activation layer and used a dropout layer to avoid overfitting that will help to generalize. Since this is regression problem we used “mean squared error” as loss function. I used Adam optimizer with the default setting. We trained the network for 10 epochs. We made sure that at least 20,000 samples are used as a training set in each epoch.  

# H1 Surprising Results-
When I used 40 K samples with a lot of recovery data and correcting the bias of steering angles, the model didn’t work. Flipping the images didn’t provide any significant help. Validation loss didn’t provide any information for the quality of the model, I am not sure why. I tried several different architectures, varying training examples, and other activation function, nothing helped.

The only thing that helped my model was the data augmentation method translation. So I retrained my model with only Udacity’s data with translation and correcting for zero bias. It worked. I used the same data augmentation technique for different architecture and activation functions, everything worked. I still don’t understand why translation is more powerful than any recovery data. Actually, adding recovery data degrades the quality of the model. 

The model performed quite well on the second track as well, except for steep turns. Adding an extra convolution layer as suggested by [Vivek Yadav]( https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee#.vjrzdttix) post and augmenting for brightness will solve those issues.
