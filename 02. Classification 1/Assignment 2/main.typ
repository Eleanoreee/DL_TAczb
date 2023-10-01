#import "template.typ": *

// Take a look at the file `template.typ` in the file panel
// to customize this template and discover how it works.
#show: project.with(
  title: "Assignment 2",
  authors: (
    "Tingyu Wu
who is DYING",
  ),
  date: "September 14, 2023",
)

// We generated the example code below so you can see how
// your document will look. Go ahead and replace it with
// your own content!

== Problem 3
Why don't use MSE as loss function for classification?

== Solution 3
AS for MSE we have: $ cal(L) = 1/m (y - hat(y))^2 $
In the problem of classification, we normally use Logistic Regression or Softmax Regression. Take Logistic Regression as an example, our fitting function will be: $ f(z) = 1/(1+e^(-z)) $

If we are going to use MSE as the loss function, it will be: $ cal(L)(w)= (y - 1/(1+e^(-(w x))))^2 $
In order to minimize this loss function above, normally we will use Gradient Descent. Let's now take derivative: $ cal(L) '(w)= (f(z)-y) f'(z)x $
// While$ 1-f(z) = e^(-z) / (1-e^(-z)) $
// Thus, $ f'(z)= f(z)(1-f(z)) $

From the above Eq we could realize that *the loss function using MSE is not convex* since the function $f(z) = 1/(1+e^(-z))$ is not convex, which will cause the problem of only finding local minimum rather than absolute minimum.

On the other hand, if we use MLE as the loss function: $ J(w) = -[y ln(f(z)) + (1-y) ln(1-f(z))] $
Take the derivative: $ J'(w) = (f(z)-y)/(f(z)(1-f(z))) f'(z)x $
For the function $f(z)$,
$ f'(z) = e^(-z)/((1+e^(-z))^2) =  f(z)(1-f(z)) $
Thus, $ J'(w) = (f(z)-y)x $

Comparing $cal(L)'(w)$ to $J'(w)$, we find out that $cal(L)'(w)$ have one more term, $f'(z)$, which reaches maximum $1/4$ when $z=0$. Therefore, *the velocity of convergence would be larger with MLE*.


== Problem 4
What's the relationship between log-odds and logistics, what's the relationship between log-odds and self-information? Interpret the result you get.
(l0g-odds is $log(p/(1-p))$)

== Solution 4
- $log(p/(1-p))$
Let's directly prove that sigmoid(log-odds)=p.
$ exp{-log(p/(1-p))} = (1-p)/p $
Hence, 
$ 1/(1+exp{-z}) = 1/(1+(1-p)/p) = p $
Therefore, we say that log-odds is the inverse function of sigmoid.

- $I = log 1/(p(x))$
$ "log-odds"(x) &= log(p/(1-p)) \
                &= log(p) - log(1-p) \
                &= I(1-p) - I(p) 
$
In this situation, log-odds can be used to show the difference between the self-information of whether an event is happened, or not.

== Problem 6
Prove KL Divergence is non-negative.

== Solution 6
According to KL Divergence:
$ D_P (Q) &= D_(K L)(Q||P) = H_P (Q) - H_Q (P) \
          &= sum_x Q(x) log 1/(P(x)) - sum_x Q(x) log 1/(Q(x)) \
          &= sum_x Q(x) log Q(x)/(P(x))
$

Jensen's Inequality tells us: for any real function  $f(x)$ which is convect on the interval $I$, the below inequality is satisfied:
$ f(sum_(i=1)^N p_i x_i) <= sum_(i=1)^N p_i f(x_i) $
while $p_i >= 0$, $sum_(i=1)^N p_i = 1$. Also, $log(x)$ is a convex function.

Thus,
$ D_P (Q) &= - sum_x Q(x) log P(x)/(Q(x)) \
          &<= - log(sum_x Q(x) P(x)/(Q(x))) \
          &= - log(sum_x P(x)) \
          &= - log(1) \
          &= 0
$