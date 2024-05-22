# How to build:

If you use nix, you can enter a dev shell:
```
nix-shell
```

Otherwise you need to install dependencies:
- cmake
- gcc13 (the C++ program uses C++20 features)
- pandas
- scipy
- matplotlib
- scienceplots

## Create a build directory and enter it:

```
mkdir build
cd build
```

## Run CMake and make

```
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Run the program

```
# On server device
./server [configFile].ssv

# On both client devices, with a 1 for the first client and a 2 for the second.
./client [configFile].ssv [1|2]

# Both server and client should use the same config file.
```

## Run the plot scripts

```
./plotter.py [pathtoprogramoutput.csv]
./theoretical_plots.py
```

The script expects a `../images/` directory to exist.

# Config file structure

The config file is just a space separated values file with two columns, the setting name and its associated value.
The name is only used for making the file more human readable, but in the actual parsing the line number determines which setting it actually is.

An example of the contents of a config file:

```
burstcount 500
burstperiod(ns) 208300
pi1svpacketsperburst 4
pi2svpacketsperburst 4
pi1ippacketsperburst 1
pi2ippacketsperburst 0
switchbuffersize(bytes) 2000000
maxlatency(ns) 156000
svpacketsize(bytes) 160
ippacketsize(bytes) 1500
```
