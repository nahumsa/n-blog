+++
title =  "Changing gates on a predifined circuit (Qiskit)"
date = 2020-09-08T20:00:00Z
author = "Nahum Sá"
+++

# Changing gates on a predefined circuit in Qiskit

This notebook was inpired by a the paper by Czarnik et al. - [Error mitigation with Clifford quantum-circuit data](https://arxiv.org/abs/2005.10189). Where you convert your arbitrary circuit into a clifford circuit that is simulable classically, a technique known as Clifford Data Regression (CDR).

So I was searching for to change a pre-existing circuit and I did not found any proper tutorial to do this on qiskit, even though it is simple, it is nontrivial and you need to change some inner objects inside the [QuantumCircuit](https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html).

# 1) Defining the circuit

First we define a circuit that we want to change all non-clifford gates to clifford gates(X, Y, Z, and CNOT). This is done in the next cell: 


```python
from qiskit import QuantumCircuit, QuantumRegister
qr = QuantumRegister(2, name='q')
qc = QuantumCircuit(qr)
qc.t(qr[0])
qc.cx(qr[0], qr[1])
qc.x(qr[0])
qc.draw('mpl')
```


![Original Circuit](/n-blog/figures/2020-09-08-Changing-Circuits_files/2020-09-08-Changing-Circuits_2_0.svg)


Now we define a second circuit which will hold our modified circuit, if you do not need to keep the previous circuit, you can skip this part and overwrite the previous circuit.


```python
qc_clifford = QuantumCircuit(2)
qc_clifford.draw('mpl')
```


![Blank Circuit](/n-blog/figures/2020-09-08-Changing-Circuits_files/2020-09-08-Changing-Circuits_4_0.svg)

<p style="text-align:center;"><img src="{{site.baseurl}}/assets/2020-09-08-Changing-Circuits_files/2020-09-08-Changing-Circuits_4_0.svg"></p>




# 2) Getting instructions from the circuit

In order to change your circuit you should get the instructions from the circuit that you want to change. This is done by iterating over the QuantumCircuit object, this will give you the instruction, quantum registers and classical registers of that instruction. The instruction is a class from the circuit library and this class has some attributes such as the gate name which we will use in order to identify gates that are not in the clifford group, but you can use other attributes of this class such as `_params` for `u3`, `u2`, and `u1` gates.


```python
import qiskit
instructions = []
gates = []
new_gates = []
for instruction, qargs, cargs in qc:
    gates.append(instruction.name)    
    if instruction.name not in ['x', 'y', 'z', 'cx']:
        instruction = qiskit.circuit.library.XGate()    
    new_gates.append(instruction.name)
    instructions.append((instruction, qargs, cargs))

print(f"Gates in the original circuit: {gates}")
print(f"Gates in the new circuit: {new_gates}")
```

    Gates in the original circuit: ['t', 'cx', 'x']
    Gates in the new circuit: ['x', 'cx', 'x']


Now we create the modified circuit by feeding the new instructions to the circuit using [`QuantumCircuit.data`](https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.data.html#qiskit.circuit.QuantumCircuit.data).


```python
qc_clifford.data = instructions
qc_clifford.draw('mpl')
```


![Clifford Circuit](/n-blog/figures/2020-09-08-Changing-Circuits_files/2020-09-08-Changing-Circuits_8_0.svg)

# References

1 -  [Qiskit Documentation](https://qiskit.org/documentation/)




---------------------------------------------------------------------------------

{{< rawhtml >}}
  <h3>Version Information</h3><table><tr><th>Qiskit Software</th><th>Version</th></tr><tr><td>Qiskit</td><td>0.19.1</td></tr><tr><td>Terra</td><td>0.14.1</td></tr><tr><td>Aer</td><td>0.5.1</td></tr><tr><td>Ignis</td><td>0.3.0</td></tr><tr><td>Aqua</td><td>0.7.0</td></tr><tr><td>IBM Q Provider</td><td>0.7.0</td></tr><tr><th>System information</th></tr><tr><td>Python</td><td>3.7.3 | packaged by conda-forge | (default, Jul  1 2019, 21:52:21) 
  [GCC 7.3.0]</td></tr><tr><td>OS</td><td>Linux</td></tr><tr><td>CPUs</td><td>2</td></tr><tr><td>Memory (Gb)</td><td>7.664028167724609</td></tr><tr><td colspan='2'>Tue Sep 08 20:12:44 2020 -03</td></tr></table>
{{< /rawhtml >}}
