#import "template.typ": *

// Take a look at the file `template.typ` in the file panel
// to customize this template and discover how it works.
#show: project.with(
  title: "Assignment 3",
  authors: (
    "Eleanore who is DYING",
  ),
  date: "September 28, 2023",
)

// We generated the example code below so you can see how
// your document will look. Go ahead and replace it with
// your own content!


== Problem 4
Demonstrating the equivalence between a multiple layer neural network without an activation function and a layer of linear networ
== Solution 4
Let's firstly consider a 2 layer neural network, while $W_1$ is weight matrix and $b$ is bias, it would compute $W_1 x + b_1$.

A second layer would then compute: $ W_2(W_1 x + b_1) + b_2 = W_2W_1 x + W_2b_1 +b_2 $
//$ W_3(W_2W_1 x + W_2b_1 +b_2) + b_3 = W_1W_2W_3x + W_2W_3b_1 + W_3b_2 + b_3 $
*which is equivalent to $W'x + b'$*.
Also, adding layers will not change the resulte. 

Thus, we can concludes that MLP without activation function is equivalent to only a layer of linear network. This also tells us the function of activation function: add non-linear properties.
//$ W_n (W_(n-1)x + b_(n-1)) +b_n = (product_(i=1)^(n) W_i) x +   $

== Problem 5
What does the negative sign signify in Gradient Descent?
== Solution 5
GD moves the vector in the *opposite direction* of the current slope towards the minima.

== Problem 6
What could be the outcome if there are too many layers with sigmoid as the activation function?
== Solution 6
Firstly, since $sigma$ is based on exponetial function, the *calculated amount is big*.

Secondly, when we use GD, the fomular for updating weight is,
$ w_(i+1) = w_i - eta (diff cal(L))/(diff w_t) $
while
$ (diff cal(L))/(diff w_i) &= (diff cal(L))/(diff x_i) dot (diff x_i)/(diff z_i) dot (diff z_i)/(diff w_i) \
&= (diff cal(L))/(diff x_i) dot sigma'(z_i)x_(i-1)
$
Since the derivative of $sigma$ is
$ sigma '(z) &= e^(-z)/((1+e^(-z))^2)
= sigma (z) (1-sigma (z))
$
Also, the range of derivative of $sigma$ is $(0, 0.25)$.

Thus, in the process of BP, as we approaching input layer, the continued multiplication will become smaller, causing *the update of gradient become slower*. In this situation, the neural network just work in shallow layers, in fact.

== Problem 7
Prove $tanh (z)+1 = 2 sigma (2z)$, and explore their potential relationship and why we replace sigmoid with tanh. (hint: range, derivative)
== Solution 7
As we know, 
$ tanh (z) &= (1-e^(-2z))/(1+e^(-2z)) \
  tanh (z) + 1 &= 2/(1+e^(-2z)) \
$
while
$ 2 sigma (2z) = 2 1/(1+e^(-2z))
$
Thus, $tanh (z)+1 = 2 sigma (2z)$

Then,let's look at the difference between this two function by graph.
#figure(
  image("tanh vs. sigmoid.png", width: 40%),
  caption: [
    the image of $tanh$ vs. sigmoid
  ],
)
Transformation from $sigma$ to $tanh$ make the center(inflection point) of activation function change from $0.5$ to $0$. Thus, *use of $tanh$ will make the probability distribution after activating centered at $0$* rather than $0.5$, which is more natural.

Then, let's find the derivatives.
$ sigma '(z) = sigma (z) (1-sigma (z)) $
$ tanh '(z) &= (4e^(-2z))/(1+e^(-2z))^2 \
&= ((1+e^(-2z))^2-(1-e^(-2z))^2)/(1+e^(-2z))^2 \
&= 1- tanh (z)^2
$

Calculating and comparing the range of derivative for $tanh$ and $sigma$, we find,
$ tanh'(z) &: (0, 0.25) \
  sigma '(z) &: (0, 1)
$
Thus, *larger derivatives of $tanh$ lead to faster convergence during training*, as updates to the model's parameters are more substantial.

//Thus, in the process of GD with backpropagation, while $z=w x$ and $y = g(z)$ the local gradient is,
//$ (diff a)/(diff w) &= (diff a)/ (diff z) dot (diff z)/(diff w) \
//&= a(1-a) dot x
//$
//Since $a(1-a)$ is always larger than $0$, the sign of local gradient will only depends on $x$. After proceesed by activation function $a= sigma(z)$, all $x$ will be larger than $0$, making the sign of the local gradient $(diff a)/(diff w)$ always positive. 


== Problem 8
How can the problem of Overfitting be solved? Provide a list of at least three methods and illustrate two of them.
== Solution 8
*1. Imporve training dataset.*
We could have find or create more data. 

*2. Randonly dropout some point in training set*
We could randomly egnore some of the neuro in the process of training.

*3. Use simple model rather than complicated one*.

== Problem 9
Thinking: Why does model training require more VRAM than inference? Not necessary to prove it, show me your guess.
== Solution 9
In the process of *training*, which is usually refers to BP. It requires space to *store each weight’s gradients and learning rates*.

*Inference* refers to FP, where only the parameters of network need to be active in the memory. The activations are discarded once the forward pass moves to a new layer. Hence, only the layer that is active in memory and the layer that gets calculated are comsumpting memory. Thus, inference only needs to continuously *hold the network parameters and temporarily hold two feature maps*.