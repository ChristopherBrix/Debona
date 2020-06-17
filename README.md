# Debona (NIPS'20, under review)
Neural networks commonly suffer from a vulnerability to adversarial attacks, where small modifications to the input cause the network to misclassify examples. In saftey-critical applications, formal verification of safety properties may be necessary to guarantee robustness against such attacks.

Debona (see [Debona: Decoupled Boundary Network Analysis for Tighter Bounds and Faster Adversarial Robustness Proofs](https://arxiv.org/abs/2006.09040)) is an improved version of [Neurify](https://github.com/tcwangshiqi-columbia/Neurify) (see [Efficient Formal Safety Analysis of Neural Networks](https://arxiv.org/abs/1809.08098)). It utilizes symbolic linear relaxation to compute tight overestimations for the bounds of the network nodes.

Compared with Neurify, Debona allows to compute proofs up to 78% faster, enabeling the analysis of deeper and more complex networks. A detailed performance comparison between Debona and Neurify is done in the paper. 



## Prerequisite

WARNING: Debona is still under active development, the git history may not be stable! If you are interested in using Debona, and need a stable version, please create an issue and we will avoid rebasing in the future. Otherwise, a stable version will be created until September.

Debona has been tested on Ubuntu 16.04.

### OpenBLAS Installation
The OpenBLAS library is used for matrix multiplication speedup. So please make sure you have successfully installed [OpenBLAS](https://www.openblas.net/). You can follow the following commands to install openblas or find the quick installation guide at [OpenBLAS's Installation Guide](https://github.com/xianyi/OpenBLAS/wiki/Installation-Guide).

```sh
wget https://github.com/xianyi/OpenBLAS/archive/v0.3.9.tar.gz
tar xzfv v0.3.9.tar.gz
cd OpenBLAS-0.3.9
make
make PREFIX=/path/to/target/library/folder install
```

### Downloading

```
git clone https://github.com/ChristopherBrix/Debona
```

### Compiling:
Please make sure the path of OpenBLAS is the same as the one in MakeFile. Then, run

```
cd Debona
make
```

## Execution
After installing the program, you can evaluate any of the 1,000 test images on any network by running 

```sh
./network_test [image_nr] [path/to/model]
```

So to test image 42 on the network `mnist50.nnet`, run
```sh
./network_test 42 models/mnist50.nnet
```

The analysis is configured to abort after 1 hour if no result is determined. 

Per default, the input is checked for a vulnerability against an attack with a perturbation of at most `L-infinite = 10`. Other bounds can be set by editing `network_test.c`.

## File Structure

* network_test.c: main file to run with
* nnet.c: deal with network instance and do symbolic interval analysis
* split.c: manage iterative refinement and dynamic thread rebalancing
* matrix.c: matrix operations supported by OpenBLAS
* models/: all the models

## Citing Debona
If you are using Debona, please cite
```
@misc{brix2020debona,
    title={Debona: Decoupled Boundary Network Analysis for Tighter Bounds and Faster Adversarial Robustness Proofs},
    author={Christopher Brix and Thomas Noll},
    year={2020},
    eprint={2006.09040},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

Debona is a fork of Neurify
```
@inproceedings {shiqi2018neurify,
	author = {Shiqi, Wang and Pei, Kexin and Justin, Whitehouse and Yang, Junfeng and Jana, Suman},
	title = {Efficient Formal Safety Analysis of Neural Networks},
	booktitle = {32nd Conference on Neural Information Processing Systems (NIPS)},
	year = {2018},
	address = {Montreal, Canada}
}
```


## Contributors

* [Christopher Brix](https://christopher-brix.de) - christopher.brix@rwth-aachen.de
* [Thomas Noll](https://moves.rwth-aachen.de/people/noll) - noll@cs.rwth-aachen.de
* [Shiqi Wang](https://sites.google.com/view/tcwangshiqi) - tcwangshiqi@cs.columbia.edu
* [Kexin Pei](https://sites.google.com/site/kexinpeisite/) - kpei@cs.columbia.edu
* [Justin Whitehouse](https://www.college.columbia.edu/node/11475) - jaw2228@columbia.edu
* [Junfeng Yang](http://www.cs.columbia.edu/~junfeng/) - junfeng@cs.columbia.edu
* [Suman Jana](http://www.cs.columbia.edu/~suman/) - suman@cs.columbia.edu


## License
Copyright (C) 2020 by its authors and contributors and their institutional affiliations under the terms of modified BSD license.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
