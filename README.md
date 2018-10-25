# Model-based, model-free Learning Simulator

The structure of the simulation is described here http://www.sanghyunyi.ml/pdf/paper/cosyne-2018.pdf

## Running simulation
The program is written and tested against python3.6, please choose appropriate python interpreter to run simulation.
```
python main.py [options]
```
To view the complete list of options, run `python main.py -h`

## Improve performance with C/C++ extension
The default running mode uses C++ extensions which need to be build manually. If no dynamic linking library found under `lib`, it will switch back to pure python execution which is around 10 times slower in total. You can also disable C++ execution manually by passing `--disable-c-ext`.
### Build
Run 
```
make -C ./lib
```
If build failed, check your library and makefile. There are some python version specific linking parameters in the makefile. The makefile is simple and you are expected to make neccessary changes in the makefile yourself.
### Platform
- Tested: Ubuntu 18.04
- Expected to work: All *nix systems
- Source code should work: Windows
### Dependencies and build requirements
- `boost::python` and some other boost libraries. So just download the whole boost library and follow their website instruction to build appropriate `boost::python` binary against your installed python interpreter.

## Known issue
- `sklearn` may have some trouble with cloudpickle, see solutions https://stackoverflow.com/questions/52596204/the-imp-module-is-deprecated
- `ggplot` may have some issue with pandas, see solutions https://stackoverflow.com/questions/50591982/importerror-cannot-import-name-timestamp