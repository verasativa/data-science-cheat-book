# Verae data science cheat-book
I'm Vera, and this is my cheat-book. If you want correct, complete or add something, don't hesitate to do a pull request. Check out a [smaller but pretty useful cheat sheet(https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-supervised-learning) I found while writing this].
## Table of content
## Clasical statistics
### Probability theory
Probability theory is the mathematical fundation for statisctics, but in contrast to probability theory statistcs is an applied science corcerned with analysis and modeling of data. Check out [Thomas Bayes](https://en.wikipedia.org/wiki/Thomas_Bayes), [Pierre-Simon Laplace](https://en.wikipedia.org/wiki/Pierre-Simon_Laplace) and [Carl Gauss](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss) for history.

#### Bayes' rule:
Bayes' rule describes the probability of an event, based on prior knowledge of conditions that might be related to the event.

$$ P(A|B) = \frac{P(B|A) P(A)}{P(B)} $$

#### Exploratory data analysis:
Check out [John Tukey](https://en.wikipedia.org/wiki/John_Tukey) for history.

Yuri Carvajal's check list:
 1. Identify the dataset:
   - What's dataset source?
   - What's dataset date/time?
   - How many records has?
   - How many pure duplicates contains?
   - What type is each column?
  2. Check codification, labels and missing values
  3. Identify modes, asymmetries, outliers, clusters, correlations and other patterns
  4. Identify statistical inconsistencies
  5. Identify technical (domain) inconsistencies
  6. Compatibilty with other users (export to csv, sql, etc)

#### Estimates of location:
 - Mean / average (the sum of all values divided by the number of values)
 - Weighted mean (the sum of all values divided by the sum of weights)
 - Median (the value that half values lies above and hafl lies belows)
 - Weighted median
 - Trimmed mean (averages after n "extreme values" are removed from top and buttom)
 - Robust (not sensitive to extreme values)
 - Outlier (values very diferent from most of data)

#### Estimates of variability:
Variability also called dispersion, measure whether the data values are tighly clustered or spread out.
 - Deviations / errors / residuals (the difference between the observed values and estimate of location)
 - Variance / mean squared error (the sum of squared deviations from mean divided by values count - 1)
 - Standard deviation / l2-norm / Euclidian norm (the squared root of variance)
 - Mean absolute deviation / l1-norm / Manhattan norm (the mean of absolute values of the deviations from the mean)
 - Mean absolute deviation from the median (the media of the absolute values of the deviations from the median)
 - Range (the difference betweeen the largest and the smallest values in data set)
 - Order statistics / ranks (Metrics based on the data values sorted from smallest to biggests)
 - Percentile / quantile (the value such P percent of the values take on this values or less and 100-P take on this value or more)
 - Interqurtile range / IQR (The diference between the 75th percentile and the 25th percentile)

#### Exploring distribution
 - Box / box and whiskers plot
 - Frecuency table and histograms
 - Density Estimates
##### Boxplot / box and whiskers plot:
A plot ntroduced by [Tukey](https://en.wikipedia.org/wiki/John_Tukey) as a quick way to visualize the distribution of data,

```python
# Pandas example
df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
df.plot.box()
```
<img src="img/box_plot_new.png">
#### Binary and categorical dataset
 - Mode (the most occurring category or value in a data set)
 - Expected value (when categories can be associated with a numerica values, this give an average value base on a category's probability of occurrence)
 - Bar charts
 - Pie Charts ([Florence Nightingale](https://en.wikipedia.org/wiki/Florence_Nightingale))

## Machine learning
Machine learning ecompases all.

<img src="img/ml_map.png">

### Simple Vector Machine
[multiclass SVM](http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/)

### Deep learning
Deep learning is any algoritm more than 1 layer of perceptrons. Input layer units = fearues, output units = labels (one if is a regression).
<img src="img/network_diagram.png">

#### Activation function:
After multiplying each imput x for it's corresponging weight, and added the bias. The perseptron shoud decide if to activate and how much, this is the job activation function. Teorically sigmoid the introductory, in practice ReLU or leakyReLU its used.
<img src="img/activation.png">

#### Feed fordward and backwards:
Feed fordward is the process of getting predictions, feeding the network with features and ending with labels / value(s) predictions.
The result of each perceptron can be noted as:
$$\hat{y} = \sigma(w_1 x_1 + w_2 x_2 + b)$$
where y hat is the output, sigma the [activation function](#Activation-function), for each input x is a weight w, and a bias b is added.
<img src="img/backprop_diagram.png">

#### Layers types:
 - Direct connected layers
 - Convolutional layers
 - Pooling layers

#### Weights inicialization:
Good practice is to start your weights randomly in the range of $$[-y, y]$ where $y=1/\sqrt{n}$$   ($$n$$ is the number of inputs to a given neuron).  Optimally random normal distribution having a mean of 0 and a standard deviation of $$y=1/\sqrt{n}$$, so most of the values are close to 0.











#### Code snipets:
##### Data loaders:

##### GPU auto:
```python
import torch

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
```

##### Define model:
```python
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (get a 32 x 32 x 3 vector in)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # convolutional layer (get a 16 x 16 x 16 vector in)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (get a 8 x 8 x 32 vector in)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)
        # Linear layer 1 (get a flated 4x4x64 vector in)
        self.linear1 = nn.Linear(4 * 4 * 64, 500)


        self.linear2 = nn.Linear(500, 10)
        #self.output_layer = nn.LogSoftmax(dim=1) # Takes 10 inputs (classes)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # flatten image input
        x = x.view(-1, 64 * 4 * 4)


        # add dropout layer
        x = self.dropout(x)
        x = F.relu(self.linear1(x))

        # add dropout layer
        x = self.dropout(x)
        x = F.log_softmax(self.linear2(x), dim=1)
        #x = F.softmax(x, dim=1)
        return x
# create a complete CNN
model = Net()
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()
```
##### Set an optimizer:
```python
import torch.optim as optim

# specify loss function
criterion = nn.NLLLoss()

# specify optimizer
optimizer = optim.LBFGS(model.parameters(), lr=1)
```
##### Training loop:
Its where the net is trained, example:
```python
# number of epochs to train the model
n_epochs = 30 # you may increase this number to train a final model

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    ###################
    # train the model #
    ###################
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        def closure():        
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            return loss

        # perform a single optimization step (parameter update)
        optimizer.step(closure)

        # calculate the batch loss
        loss = criterion(output, target)
        # update training loss
        train_loss += loss.item()*data.size(0)

    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss
        valid_loss += loss.item()*data.size(0)

    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = valid_loss
```
