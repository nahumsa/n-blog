+++
title = "Quantum State Tomography"
date = 2020-06-22T08:00:00Z
name = "Nahum Sá"
+++


# 1) Introduction

Let's try to do the tomography of a given one-qubit quantum state just to understand the principle behind this procedure:

$$
    \rho = \begin{pmatrix}
                \rho_{11} & \rho_{12} \\\\
                \rho_{21} & \rho_{22}
           \end{pmatrix} =  \frac{1}{2} \sum_{i=0}^3 S_i \sigma_i
$$

Where  $\rho_{11} + \rho_{22} = 1$, $\rho_{12} = \rho_{21}^*$, and $\sigma_i$ are the pauli matrices. 

$$
\sigma_0 = I \ \ \ \sigma_1 = \sigma_x = \begin{pmatrix} 0 & 1 \\\\ 1 & 0 \end{pmatrix} \ \ \ \sigma_2 = \sigma_y = \begin{pmatrix} 0 & i \\\\ -i & 0 \end{pmatrix} \ \ \ \sigma_3 = \sigma_z = \begin{pmatrix} 1 & 0 \\\\ 0 & -1 \end{pmatrix}
$$

Our goal is to find $S_i$ an then characterize the state.

A problem is that the IBM quantum computers only measures on the Z axis, therefore we only can know $S_3$ that is associated as the expected value of the measurement along the Z axis. 
We need to do a trick to be able to measure $\big< X \big>$ and $\big< Y \big>$.

Let's first check what are the eigenvectors of those operators.

- X has as eigenvectors $\left| + \right> = \frac{1}{\sqrt{2}} \bigg( \left| 0 \right> + \left| 1 \right> \bigg)$ 
and $\left| - \right> = \frac{1}{\sqrt{2}} \bigg( \left| 0 \right> - \left| 1 \right> \bigg)$ ;

- Y has as eigenvectors $\left| i^+ \right> = \frac{1}{\sqrt{2}} \bigg( \left| 0 \right> + i\left| 1 \right> \bigg)$ 
and $\left| i^- \right> = \frac{1}{\sqrt{2}} \bigg( \left| 0 \right> - i\left| 1 \right> \bigg)$ ;


Thus to measure on those bases we need only to add a way to generate those eigenvectors on the circuit:

- To generate X, we only need to apply the gate $H$. 
- To generate Y, we need to apply the gate $S^{\dagger}H$.


# 2) Doing Quantum State Tomography by hand on qiskit

```python
import qiskit as qsk
import numpy as np
import matplotlib.pyplot as plt
```

In order to test this, consider we have the state $\left| \psi \right> = \frac{1}{\sqrt{2}} \big( \left| 0 \right> + \left| 1 \right> \big)$ and we want to do the tomography of this state.


```python
def measure_X(circuit,n):    
    circuit.barrier(n)
    circuit.h(n)
    circuit.measure(n,n)
    return circuit

def measure_Y(circuit,n):    
    circuit.barrier(n)
    circuit.sdg(n)
    circuit.h(n)    
    circuit.measure(n,n)    
    return circuit

def tomography(circuit):
    """ Tomography of a one qubit Circuit.
    """
    qc_list = []
    base = ['X', 'Y', 'Z']
    for basis in base:
        Q = qsk.QuantumCircuit(1,1)
        Q.append(circuit, [0])
        if basis == 'X':
            measure_X(Q, 0)
            qc_list.append(Q)
        if basis == 'Y':
            measure_Y(Q,  0)
            qc_list.append(Q)
        if basis == 'Z':
            Q.measure(0,0)
            qc_list.append(Q)
    return qc_list, base
```


```python
qc = qsk.QuantumCircuit(1)
qc.h(0)
qcs, bases = tomography(qc)
```

Running tomography


```python
backend_sim = qsk.Aer.get_backend('qasm_simulator')
job = qsk.execute(qcs, backend_sim, shots=5000)
result = job.result()

for index, circuit in enumerate(qcs):
    print(result.get_counts(circuit))
    print(f'Base measured {bases[index]}\n')
```

    {'0': 5000}
    Base measured X
    
    {'0': 2484, '1': 2516}
    Base measured Y
    
    {'0': 2503, '1': 2497}
    Base measured Z
    
Thus we have: X = 1, Y = 0 and Z = 0, because 0 is equivalent to +1 and 1 equivalent to -1. It is not exactly 0 because of statistical fluctiations.



```python
def get_density_matrix(measurements,circuits):
    """Get density matrix from tomography measurements.

    """
    density_matrix = np.eye(2, dtype=np.complex128)
    sigma_x = np.array([[0,1],[1,0]])
    sigma_y = np.array([[0,-1j],[1j,0]])
    sigma_z = np.array([[1,0],[0,-1]])
    basis = [sigma_x, sigma_y, sigma_z]

    for index in range(len(circuits)):
        R = measurements.get_counts(index)
        
        if '0' in R.keys() and '1' in R.keys():
            zero = R['0']
            one = R['1']
        
        elif '1' in R.keys():
            zero = 0
            one = R['1']
        
        elif '0' in R.keys():
            zero = R['0']
            one = 0

        total = sum(list(R.values()))
        expected = (zero - one)/total        
        density_matrix += expected * basis[index]

    return 0.5*density_matrix

density = get_density_matrix(result,qcs)
print(density)
```

    [[0.5006+0.j     0.5   +0.0032j]
     [0.5   -0.0032j 0.4994+0.j    ]]

Writing the density matrix:
$$
    \rho = \frac{1}{2} \begin{pmatrix} 
            1 & 1 \\\\
            1 & 1
           \end{pmatrix}
$$



# 3) Doing Quantum State Tomography using Ignis

Qiskit has a module inside ignis to do [tomography](https://qiskit-staging.mybluemix.net/documentation/ignis/tomography.html) that does exactly what we've done before, but is generalized for multiple qubits as input.


```python
from qiskit.ignis.verification.tomography import state_tomography_circuits

tomography_circuits = state_tomography_circuits(qc, [0], meas_labels='Pauli', meas_basis='Pauli')
```

As a check, let's see how they measure the Y basis, and we can verify that it is the same way as we did before.


```python
tomography_circuits[1].draw('mpl')
```

![tomography circuit](/n-blog/figures/2020-06-22-Tomography_files/2020-06-22-Tomography_14_0.svg)

```python
backend_sim = qsk.Aer.get_backend('qasm_simulator')
job = qsk.execute(tomography_circuits, backend_sim, shots=5000)
result = job.result()
```

They also has a function to transform the results to a density matrix which treats the state tomography as an optimization problem. The `fit` method has several methods for fitting the tomography results, such as `cvx` which is convex optimization and `lstsq` which makes uses the least square optimization.


```python
from qiskit.ignis.verification.tomography import StateTomographyFitter

state_fitter = StateTomographyFitter(result, tomography_circuits, meas_basis='Pauli')
```


```python
density_matrix = state_fitter.fit(method='lstsq')
print(density_matrix)
```

    [[0.49040221+0.j         0.49988484+0.00479889j]
     [0.49988484-0.00479889j 0.50959779+0.j        ]]


Which is approximately what we expect, as shown before.


# 4) State tomography of 3 qubits

For this example, let's use the GHZ state:

$$
    \left| \psi \right> = \frac{1}{\sqrt{2}} \bigg( \left| 000 \right> + \left| 111 \right> \bigg)
$$

This state is simply constructed using an H gate and two CNOT gates:

```python
qc = qsk.QuantumCircuit(3)
qc.h(0)
qc.cx(0,1)
qc.cx(0,2)
qc.draw('mpl')
```

![tomography circuit](/n-blog/figures/2020-06-22-Tomography_files/2020-06-22-Tomography_22_0.svg)

In order to indicate which qubits we want to do tomography, we need to put an list after choosing the circuit that we want to do tomography. In this case, we want to do tomography of all qubits of qc, therefore we need to put `[0,1,2]`.


```python
from qiskit.ignis.verification.tomography import state_tomography_circuits

tomography_circuits = state_tomography_circuits(qc, [0,1,2], 
                                                meas_labels='Pauli', 
                                                meas_basis='Pauli')
backend_sim = qsk.Aer.get_backend('qasm_simulator')
job = qsk.execute(tomography_circuits, backend_sim, shots=5000)
result = job.result()
```


```python
from qiskit.ignis.verification.tomography import StateTomographyFitter

state_fitter = StateTomographyFitter(result, tomography_circuits, meas_basis='Pauli')
```


```python
density_matrix = state_fitter.fit(method='lstsq')
```

Let's define a helper function that plots the real and imaginary part as a grey scale image:

```python
def plot_density_matrix(DM):
    """Helper function to plot density matrices.

    Parameters
    ----------------------------------------------
    DM(np.array): Density Matrix.
    
    """
    from matplotlib.ticker import MaxNLocator
    
    fig = plt.figure(figsize=(16,10))
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    im = ax1.imshow(np.real(DM), cmap='Greys')
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_title('Real Part',size=16)
    plt.colorbar(im, ax=ax1)

    im = ax2.imshow(np.imag(DM), cmap='Greys')
    plt.colorbar(im, ax=ax2)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_title('Imaginary Part',size=16)
    plt.show()
```

Ploting the density matrix, we see that there are states only on the first and last column (row), each column (row) represents a quantum state on the computational basis, for this case the states represented are $\left| 000 \right>$ and $\left| 111 \right>$ respectively. We can ignore the imaginary part, since is is only random noise.

```python
plot_density_matrix(density_matrix)
```
![Density matrix](/n-blog/figures/2020-06-22-Tomography_files/2020-06-22-Tomography_26_0.svg)

The code for this post is on [github](https://github.com/nahumsa/Introduction-to-IBM_Qiskit/blob/master/Notebooks/Tomography.ipynb).

# References 

1 - [Altepeter et al - Quantum State Tomography](http://research.physics.illinois.edu/QI/Photonics/tomography-files/tomo_chapter_2004.pdf)

2 - [Qiskit Ignis Documentation](https://qiskit.org/documentation/apidoc/verification.html#tomography)

-------------------------------------------------------------------------------------------------
{{< rawhtml >}}
<h3>Version Information</h3><table><tr><th>Qiskit Software</th><th>Version</th></tr><tr><td>Qiskit</td><td>0.19.1</td></tr><tr><td>Terra</td><td>0.14.1</td></tr><tr><td>Aer</td><td>0.5.1</td></tr><tr><td>Ignis</td><td>0.3.0</td></tr><tr><td>Aqua</td><td>0.7.0</td></tr><tr><td>IBM Q Provider</td><td>0.7.0</td></tr><tr><th>System information</th></tr><tr><td>Python</td><td>3.7.3 | packaged by conda-forge | (default, Jul  1 2019, 21:52:21) 
[GCC 7.3.0]</td></tr><tr><td>OS</td><td>Linux</td></tr><tr><td>CPUs</td><td>2</td></tr><tr><td>Memory (Gb)</td><td>7.664028167724609</td></tr><tr><td colspan='2'>Sat May 23 17:14:48 2020 -03</td></tr></table>
{{< /rawhtml >}}
