# Model-based, model-free Learning Simulator

The structure of the simulation is described here http://www.sanghyunyi.ml/pdf/paper/cosyne-2018.pdf

## System requirements and required libraries
Tested Platform: 
- Ubuntu 18.04 LTS
- GCC 7.3.0
- Clang 6.0.0
- Python: 3.6.7

Required Python Packages:

| Package Name | Tested Version |
| ------------ | ------- |
| [Pytorch](https://pytorch.org/) | 0.4.1 |
| [dill](https://pypi.org/project/dill/) | 0.2.8.2 |
| [numpy](http://www.numpy.org/) | 1.15.1 |
| [matplotlib](https://matplotlib.org/) | 2.2.3 |
| [scipy](https://www.scipy.org/) | 1.1.0 |
| [scikit-learn](https://scikit-learn.org/) | 0.20.0 |
| [tqdm](https://tqdm.github.io/) | 4.26.0 |
| [gym](https://gym.openai.com/) | 0.10.5 |
| [ggplot](http://ggplot.yhathq.com/) | 0.11.5 |
| [pandas](https://pandas.pydata.org/) | 0.23.4 |

One should be able to run the script (see Running simulation section) once these packages installed. We also provided a C++ library to improve the performance. Its only dependency is `boost::python` library. If you already have `boost::python` installed, simply run
```
make -C ./lib
```
It is recommended to build `boost` library following the [official website](https://www.boost.org/doc/libs/1_67_0/more/getting_started/unix-variants.html) instructions. But I will give a specific instructions here which may be too specific that will fail depending on your machine configuration.
1. Download Boost library source code https://www.boost.org/. I am using 1.67 version. [Direct link](https://sourceforge.net/projects/boost/files/boost/1.67.0/boost_1_67_0.tar.gz/download). Extract the archive and save it to ~/Downloads/
2. Copy header files
    ```
    sudo cp -r ~/Downloads/boost_1_67_0/boost /usr/local/include/boost/
    ```
3. Configure which python interpreter to build against
    ```
    echo "using python : 3.6 : /usr/bin/python3 : /usr/include/python3.6m : /usr/lib ;" > ~/user-config.jam
    ```
4. Build Boost libraries
    ```
    cd ~/Downloads/boost_1_67_0/
    ./bootstrap.sh –prefix=/usr/local
    ./b2
    sudo ./b2 install
    ```
5. Something to check before building our C++ FORWARD extension
    - Run `ldconfig -v 2>/dev/null | grep -v ^$'\t'` to see your linking path contains `/usr/local/lib`
    - Run `echo | gcc -E -Wp,-v –` to see if your include path contains `/usr/local/include`
    - Check if you have `clang` installed. If not you can install it using apt or change variable `$(CC)` from `clang++` to `g++` in `lib/Makefile`

`boost::python` bulid complete. You can build the library now.

## Running simulation
```
python main.py [options]
```
A recommended simulation configuration
```
python main.py -d --episodes 1000 --trials 20 --all-mode --disable-detail-plot
```
Generate learning curve plots
```
python main.py --re-analysis [pickle file name] --learning-curve-plot --disable-action-compare --use-confidence-interval
```
Generate P_mf probability density function plots
```
python main.py --re-analysis [pickle file name] --cross-compare
```
To view the complete list of options, run
```
python main.py -h
```

## Known issue
- `sklearn` may have some trouble with cloudpickle, see solutions https://stackoverflow.com/questions/52596204/the-imp-module-is-deprecated
- `ggplot` may have some issue with pandas, see solutions https://stackoverflow.com/questions/50591982/importerror-cannot-import-name-timestamp