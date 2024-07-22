[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-3916/)
[![license](https://img.shields.io/badge/license-apache_2.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)

## G-DFL: Combining Gradient Information and Primitive Directions for High-Performance Mixed-Integer Optimization

Implementation of the G-DFL algorithm proposed in 

[Lapucci M., Liuzzi G., Lucidi S. & Mansueto P., Combining Gradient Information and Primitive Directions for High-Performance Mixed-Integer Optimization. arXiv pre-print (2024)](
https://arxiv.org/abs/2407.14416)

If you have used our code for research purposes, please cite the publication mentioned above.
For the sake of simplicity, we provide the Bibtex format:

```
@misc{lapucci2024combininggradientinformationprimitive,
    title={Combining Gradient Information and Primitive Directions for High-Performance Mixed-Integer Optimization}, 
    author={Matteo Lapucci and Giampaolo Liuzzi and Stefano Lucidi and Pierluigi Mansueto},
    year={2024},
    eprint={2407.14416},
    archivePrefix={arXiv},
    primaryClass={math.OC},
    url={https://arxiv.org/abs/2407.14416}, 
}
```

### Main Dependencies Installation

In order to execute the code, you need an [Anaconda](https://www.anaconda.com/) environment.

#### Main Packages

* ```python v3.9.16```
* ```numpy v1.24.3```
* ```scipy v1.10.1```
* ```tensorflow v2.13.0```

### Usage

In ```args_parser.py``` you can find all the possible arguments. Given a terminal (Anaconda Prompt for Windows users), two examples of execution could be the following.

``` python main.py --seeds 16007 --verbose --max_time 2 --type_gradient_related_direction lbfgs ```

``` python main.py --seeds 16007 --save_logs --max_time 2 --type_gradient_related_direction lbfgs ```

In the second case, the logs are saved in the ```Outputs``` folder. In ```main.py```, you can find all the documentation about how they are stored.

### Contact

If you have any question, feel free to contact us:

[Pierluigi Mansueto](https://webgol.dinfo.unifi.it/pierluigi-mansueto/)<br>
Global Optimization Laboratory ([GOL](https://webgol.dinfo.unifi.it/))<br>
University of Florence<br>
Email: pierluigi dot mansueto at unifi dot it

[Giampaolo Liuzzi](https://sites.google.com/diag.uniroma1.it/giampaolo-liuzzi/home)<br>
Dipartimento di Ingegneria Informatica Automatica e Gestionale "Antonio Ruberti" (DIAG) ([DIAG](http://www.diag.uniroma1.it/))<br>
University of Rome "La Sapienza"<br>
Email: liuzzi at diag dot uniroma1 dot it