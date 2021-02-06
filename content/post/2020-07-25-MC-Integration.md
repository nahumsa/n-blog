+++
title =  "Monte Carlo Integration"
date = 2020-07-25T17:00:00Z
author = "Nahum Sá"
+++

{{< rawhtml >}} <a href="https://colab.research.google.com/github/nahumsa/Self-Avoiding-Walks/blob/master/Monte%20Carlo%20Integration.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>{{< /rawhtml >}}


## 1) Introduction
In this blog post I will explain how to integrate an arbitrary function using Monte Carlo Integration basically we are shooting darts into a dartboard and accepting them according to a given criteria, we can represent this by the following gif:

![GIF MC integration](/figures/2020-07-25-MC-Integration_files/2020-07-25-MC-Integration.gif)

## 2) Algorithm


In Monte Carlo Integration we sample from an interval $$\{a,b\}$$ and see if it is inside the region between the function and the x axis, if this condition is satisfied we accept the sample, otherwise we discart the sample.

So basically we are shooting darts randomly and accepting them if they are inside the area bellow the function that you wish to integrate and the x axis, the mean count of on target darts are multiplied by the area that covers a paralelogram made by the max of your function inside this interval and the size of the interval. The following is the mathematical way that we

Consider that you want to estimate the following integral:

$$
\int_a^b f(x) dx
$$

The procedure can be summarized as follows: 

- 1) Draw a uniform sample from the interval $$x \sim \{a,b\}$$ and a uniform sample from $$y \sim \{0, \max (f(\{a,b\})\}$$, where $$\max (f(\{a,b\})$$ is the maximum value of the function inside the interval $$\{a, b\}$$ ;

- 2) Evaluate $$f(x)$$ and if $$f(x) > y$$ discard the sample, otherwise accept the sample.

On average you will have the number of samples that satisfies your constraints, then you take the average and multiply for the area of your given interval:

$$
A = (\max (f(\{a,b\})\} - 0)*(b-a)
$$

Thus:

$$
\int_a^b f(x) dx = A* \mathbb{E}(\mathrm{Accepted \ counts})
$$

Let's program this algorithm in python!

``` python
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
```


``` python
def mc_integration(x_init, x_final, func, n=100000):
  """ Function to do monte carlo integration for 
      n samples.

  Parameters
  -----------------------------------------------
  x_init(float): Starting point of integration.
  x_final(float): Ending point of integration.
  func(function): Python function that you want to integrate.
  n(int): Number of samples.

  """

  X = np.linspace(x_init, x_final, 1000)
  y1 = 0
  # Overshoot by 1 for convergence
  y2 = max((func(X))) + 1
      
  area = (x_final-x_init)*(y2-y1)
  check = []
  xs = []
  ys = []

  for _ in range(n):
      
      # Generate Samples
      x = np.random.uniform(x_init,x_final,1)
      xs.append(float(x))
      
      y = np.random.uniform(y1,y2,1)
      ys.append(float(y))
      
      # Reject
      if abs(y) > abs(func(x)) or y<0:
          check.append(0)
      
      # Accept
      else:
          check.append(1)

  return np.mean(check)*area, xs, ys, check
```

## Application 1: $$f(x) = \sin x$$
Let's try with a simple function: 

$$
f(x) = \sin x
$$

And compare with the integration used on scipy.


``` python

def f(x):
  return np.sin(x)
```


``` python
from scipy.integrate import quad

a = 0.3
b = 2.5

sol, xs, ys, check = mc_integration(a, b, f, n=1000000)
id_sol, _ = quad(f, a, b)

print(f'Monte Carlo Solution: {sol}')
print(f'Quad Solution: {id_sol}')
print(f'Error: {np.square(sol - id_sol)}')
```

    Monte Carlo Solution: 1.7552919930712214
    Quad Solution: 1.7564801046725398
    Error: 1.4116091771873149e-06


Let's see what is happening when we are doing this procedure, the red dots are rejected and blue dots are accepted and we have the plot of our function in blue.

We see that we sampled a lot of points in this interval such that we almost filled the area.


``` python
check = np.array(check)
xs = np.array(xs)
ys = np.array(ys)
plt.plot(xs[check == 0], ys[check == 0], 'ro', label='Rejected')
plt.plot(xs[check == 1], ys[check == 1], 'go', label='Accepted')
x = np.linspace(a,b, 100000)

plt.plot(x, f(x), label='Function', linewidth=6)
plt.legend()
plt.show()
```
![Function approximation](/figures/2020-07-25-MC-Integration_files/2020-07-25-MC-Integration_8_0.png)

Let's see how the convergence of our method is affected by the sample size.


``` python
err = []

n = np.linspace(10, 10000, 100)

for i in tqdm(n):  
  sol, *_ = mc_integration(a, b, f, n=int(i))
  err.append(np.square(sol - id_sol))
```


``` python
plt.plot(n, err)
plt.title("Convergence", size=20)
plt.xlabel("Number of Samples", size=16)
plt.ylabel("Error", size=16)
plt.show()
```
![Convergence](/figures/2020-07-25-MC-Integration_files/2020-07-25-MC-Integration_11_0.png)

## Application 2: $$f(x) = \frac{\sin x}{x^2}$$

``` python
def f(x):
  return np.sin(x)/(x**2)
```


``` python
from scipy.integrate import quad

a = 0.3
b = 5

sol, xs, ys, check = mc_integration(a, b, f, n=1000000)
id_sol, _ = quad(f, a, b)

print(f'Monte Carlo Solution: {sol}')
print(f'Quad Solution: {id_sol}')
print(f'Error: {np.square(sol - id_sol)}')
```

    Monte Carlo Solution: 1.773632401526077
    Quad Solution: 1.635995393784945
    Error: 0.01894394589993242


Let's see what is happening when we are doing this procedure, the red dots are rejected and blue dots are accepted and we have the plot of our function in blue.

We see that we sampled a lot of points in this interval such that we almost filled the area.


``` python
check = np.array(check)
xs = np.array(xs)
ys = np.array(ys)
plt.plot(xs[check == 0], ys[check == 0], 'ro', label='Rejected')
plt.plot(xs[check == 1], ys[check == 1], 'go', label='Accepted')
x = np.linspace(a, b, 100000)

plt.plot(x, f(x), label='Function', linewidth=6)
plt.legend()
plt.show()
```
![Approximation](/figures/2020-07-25-MC-Integration_files/2020-07-25-MC-Integration_16_0.png)

As before, let's see the convergence of our method:


``` python
err = []

n = np.linspace(10, 10000, 100)

for i in tqdm(n):  
  sol, *_ = mc_integration(a, b, f, n=int(i))
  err.append(np.square(sol - id_sol))
```

    100%|██████████| 100/100 [00:06<00:00, 15.33it/s]



``` python
plt.plot(n, err)
plt.title("Convergence", size=20)
plt.xlabel("Number of Samples", size=16)
plt.ylabel("Error", size=16)
plt.show()
```
![Convergence](/figures/2020-07-25-MC-Integration_files/2020-07-25-MC-Integration_19_0.png)

# Conclusion

By learning the method of Monte Carlo Integration you have learned a powerful tool that can be generalized for other algorithms such as Sequential Monte Carlo(SMC) and Markov Chain Monte Carlo(MCMC), which (hopefully) I will cover in future blog posts.
