import getopt
import sys
import csv
import simulation as sim

usage_str = """
Model-free, model-based learning simulation

Usage:
    -d load parameters from csv file, default is regdata.csv
    --set-param-file [parameter file]             Specify the parameter csv file to load
    --mdp-stages [MDP environment stages]         Specify how many stages in MDP environment
    --ctrl-mode <min-spe/max-spe/min-rpe/max-rpe> Choose control agent mode
    --disable-control                             Disable control agents
"""

def usage():
    print(usage_str)

LOAD_PARAM_FILE = False
PARAMETER_FILE  = 'regdata.csv'

if __name__ == '__main__':
    short_opt = "hd"
    long_opt  = ["help", "mdp-stages=", "disable-control", "ctrl-mode=", "set-param-file="]
    try:
        opts, args = getopt.getopt(sys.argv[1:], short_opt, long_opt)
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    for o, a in opts:
        if o == "--disable-control":
            sim.CTRL_AGENTS_ENABLED = False
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o == "--mdp-stages":
            sim.MDP_STAGES = int(a)
        elif o == "--ctrl-mode":
            sim.CONTROL_MODE = a
        elif o == "-d":
            LOAD_PARAM_FILE = True
        elif o == "--set-param-file":
            PARAMETER_FILE = a
        else:
            assert False, "unhandled option"

    if LOAD_PARAM_FILE:
        with open(PARAMETER_FILE) as f:
            csv_parser = csv.reader(f)
            for row in csv_parser:
                row = list(map(float, row[:-1])) # convert to float
                sim.simulation(row[0], row[1], row[2],
                               row[3], row[4], row[5])
    else:
        sim.simulation()