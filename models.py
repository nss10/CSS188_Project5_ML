import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.get_weights(),x)

    def get_prediction(self, x):  
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        dotProduct = self.run(x)
        return 1 if nn.as_scalar(dotProduct) >=0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        shouldTry = True
        while(shouldTry):
            shouldTry = False
            for x,y in dataset.iterate_once(1):
                if(nn.as_scalar(y) != self.get_prediction(x)):
                    self.w.update(x,nn.as_scalar(y))
                    shouldTry=True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_layer_size_list = [10,15,10]
        self.input_size = 1
        self.output_size = 1
        self.batch_size = 2
        self.param_list=[]
        for layer_size in self.hidden_layer_size_list:
            self.add_hidden_layer(layer_size)
            
        self.add_hidden_layer(self.output_size)

    def add_hidden_layer(self,layer_size):
        if(len(self.param_list)>0):
            prev_layer_size = self.param_list[-1].data.shape[1]
        else:
            prev_layer_size = self.input_size
        print(prev_layer_size,layer_size)
        self.param_list.append(nn.Parameter(prev_layer_size,layer_size))
        self.param_list.append(nn.Parameter(1,layer_size))

    def ReLU_output(self,x,W,b):
        return (nn.ReLU(nn.AddBias(nn.Linear(x,W),b)))
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        h=x
        for i in range(len(self.param_list)):
            '''Running logic the hidden layers'''
            if(i%2==0 and i < len(self.param_list)-2):
                h =  self.ReLU_output(h,self.param_list[i],self.param_list[i+1])
        
        '''For output layer'''
        return nn.AddBias(nn.Linear(h,self.param_list[-2]),self.param_list[-1])



    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        alpha = -0.005
        convergent = False
        iter=0
        while(not convergent):
            final_loss = 0
            iter+=1
            for x,y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x,y)
                grad_param_list = nn.gradients(loss, self.param_list)
                for i in range(len(self.param_list)):
                    self.param_list[i].update(grad_param_list[i],alpha)
                
            final_loss = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))  # Convergence condition ripped off from autograder.py
            if(nn.as_scalar(final_loss)<0.02):
                print(str(iter)+ "-  " + str(nn.as_scalar(final_loss)))
                convergent = True





class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        from datetime import datetime
        print("Starting at ",datetime.now().strftime('%H:%M:%S'))
        self.hidden_layer_size_list = [600,400,256,64,16]
        self.input_size = 784
        self.output_size = 10
        self.batch_size = 50
        self.param_list=[]
        for layer_size in self.hidden_layer_size_list:
            self.add_hidden_layer(layer_size)
            
        self.add_hidden_layer(self.output_size)

    def add_hidden_layer(self,layer_size):
        if(len(self.param_list)>0):
            prev_layer_size = self.param_list[-1].data.shape[1]
        else:
            prev_layer_size = self.input_size
        print(prev_layer_size,layer_size)
        self.param_list.append(nn.Parameter(prev_layer_size,layer_size))
        self.param_list.append(nn.Parameter(1,layer_size))

    def ReLU_output(self,x,W,b):
        return (nn.ReLU(nn.AddBias(nn.Linear(x,W),b)))

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h=x
        for i in range(len(self.param_list)):
            '''Running logic the hidden layers'''
            if(i%2==0 and i < len(self.param_list)-2):
                h =  self.ReLU_output(h,self.param_list[i],self.param_list[i+1])
        
        '''For output layer'''
        return nn.AddBias(nn.Linear(h,self.param_list[-2]),self.param_list[-1])
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        alpha = -0.01
        convergent = False
        iter=0
        while(not convergent):
            validation_accuracy = 0
            iter+=1
            for x,y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x,y)
                grad_param_list = nn.gradients(loss, self.param_list)
                for i in range(len(self.param_list)):
                    self.param_list[i].update(grad_param_list[i],alpha)
                
            validation_accuracy = dataset.get_validation_accuracy()
            print(str(iter)+ "-  " + str(validation_accuracy))
            if(validation_accuracy>0.975):
                convergent = True

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
