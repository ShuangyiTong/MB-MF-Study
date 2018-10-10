import getopt
import sys
import csv
import simulation as sim

from analysis import gData, MODE_MAP

usage_str = """
Model-free, model-based learning simulation

Usage:
    -d load parameters from csv file, default is regdata.csv
    -n [number of parameters entries to simulate]
    --set-param-file [parameter file]             Specify the parameter csv file to load
    --mdp-stages [MDP environment stages]         Specify how many stages in MDP environment
    --ctrl-mode <min-spe/max-spe/min-rpe/max-rpe/min-mf-rel/max-mf-rel/min-mb-rel/max-mb-rel> 
                                                  Choose control agent mode
    --disable-control                             Disable control agents
    --trials [num trials per episodes]
    --episodes [num episodes]
    --all-mode                                    Execute all control mode
    --disable-static-control                      Use DDQN control instead of static control
"""

def usage():
    print(usage_str)

LOAD_PARAM_FILE   = False
NUM_PARAMETER_SET = 22
ALL_MODE          = False
PARAMETER_FILE    = 'regdata.csv'

if __name__ == '__main__':
    short_opt = "hdn:"
    long_opt  = ["help", "mdp-stages=", "disable-control", "ctrl-mode=", "set-param-file=", "trials=", "episodes=", "all-mode", "disable-static-control"]
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
        elif o == "--episodes":
            sim.TOTAL_EPISODES = int(a)
        elif o == "--trials":
            sim.TRIALS_PER_SESSION = int(a)
        elif o == "--all-mode":
            ALL_MODE = True
        elif o == "-n":
            NUM_PARAMETER_SET = int(a)
        elif o == "--disable-static-control":
            sim.STATIC_CONTROL_AGENT = False
        else:
            assert False, "unhandled option"

    if LOAD_PARAM_FILE:
        with open(PARAMETER_FILE) as f:
            csv_parser = csv.reader(f)
            param_list = []
            for row in csv_parser:
                param_list.append(tuple(map(float, row[:-1])))
        if ALL_MODE:
            for mode, _ in MODE_MAP.items():
                sim.CONTROL_MODE = mode
                print('Running mode: ' + mode)
                for index in range(NUM_PARAMETER_SET):
                    print('Parameter set: ' + str(param_list[index]))
                    sim.simulation(*(param_list[index]), PARAMETER_SET=str(index))
                gData.generate_summary(sim.CONTROL_MODE)
        else:                
            for index in range(NUM_PARAMETER_SET):
                sim.simulation(*(param_list[index]), PARAMETER_SET=str(index))
            gData.generate_summary(sim.CONTROL_MODE)
    elif ALL_MODE:
        for mode, _ in MODE_MAP.items():
            sim.CONTROL_MODE = mode
            sim.simulation()
        gData.generate_summary(sim.CONTROL_MODE)
    else:
        sim.simulation()
        gData.generate_summary(sim.CONTROL_MODE)