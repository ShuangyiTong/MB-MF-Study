import getopt
import sys
import csv
import simulation as sim
import dill as pickle # see https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions

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
    --all-mode                                    Execute all control mode
    --disable-static-control                      Use DDQN control instead of static control
    --disable-detail-plot                         Disable plot for each simulation
    --disable-c-ext                               Disable using C extension
    --less-control-input                          Less environment input for control agent
    --re-analysis [analysis object pickle file]   Re-run analysis functions
    --episodes [num episodes]                     
    --trials [num trials per episodes]
"""

def usage():
    print(usage_str)

LOAD_PARAM_FILE   = False
NUM_PARAMETER_SET = 22
ALL_MODE          = False
ANALYSIS_OBJ      = None
PARAMETER_FILE    = 'regdata.csv'

if __name__ == '__main__':
    short_opt = "hdn:"
    long_opt  = ["help", "mdp-stages=", "disable-control", "ctrl-mode=", "set-param-file=", "trials=", "episodes=", "all-mode", "disable-static-control",
                 "disable-c-ext", "disable-detail-plot", "less-control-input", "re-analysis="]
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
            sim.TRIALS_PER_EPISODE = int(a)
        elif o == "--all-mode":
            ALL_MODE = True
        elif o == "-n":
            NUM_PARAMETER_SET = int(a)
        elif o == "--disable-static-control":
            sim.STATIC_CONTROL_AGENT = False
        elif o == "--disable-c-ext":
            sim.DISABLE_C_EXTENSION = True
        elif o == "--disable-detail-plot":
            sim.ENABLE_PLOT = False
        elif o == "--less-control-input":
            sim.MORE_CONTROL_INPUT = False
        elif o == "--re-analysis":
            ANALYSIS_OBJ = a
        else:
            assert False, "unhandled option"

    if ANALYSIS_OBJ is not None:
        with open(ANALYSIS_OBJ, 'rb') as pkl_file:
            gData = pickle.load(pkl_file)
        for mode, _ in MODE_MAP.items():
            try:
                gData.set_current_mode(mode)
                gData.generate_summary(mode)
            except KeyError:
                print('mode: ' + mode + ' data not found. Skip')
        exit(0)

    gData.trial_separation = sim.TRIALS_PER_EPISODE
    if LOAD_PARAM_FILE:
        with open(PARAMETER_FILE) as f:
            csv_parser = csv.reader(f)
            param_list = []
            for row in csv_parser:
                param_list.append(tuple(map(float, row[:-1])))
        if ALL_MODE:
            for mode, _ in MODE_MAP.items():
                gData.new_mode(sim.CONTROL_MODE)
                sim.CONTROL_MODE = mode
                print('Running mode: ' + mode)
                for index in range(NUM_PARAMETER_SET):
                    print('Parameter set: ' + str(param_list[index]))
                    sim.simulation(*(param_list[index]), PARAMETER_SET=str(index))
                gData.generate_summary(sim.CONTROL_MODE)
                gData.save_mode(sim.CONTROL_MODE)
        else:
            gData.new_mode(sim.CONTROL_MODE)
            for index in range(NUM_PARAMETER_SET):
                sim.simulation(*(param_list[index]), PARAMETER_SET=str(index))
            gData.generate_summary(sim.CONTROL_MODE)
            gData.save_mode(sim.CONTROL_MODE)
    elif ALL_MODE:
        for mode, _ in MODE_MAP.items():
            sim.CONTROL_MODE = mode
            gData.new_mode(sim.CONTROL_MODE)
            sim.simulation()
            gData.save_mode(sim.CONTROL_MODE)
    else:
        gData.new_mode(sim.CONTROL_MODE)
        sim.simulation()
        gData.save_mode(sim.CONTROL_MODE)
    
    # Save the whole analysis object for future reference
    with open(gData.file_name('Analysis-Object') + '.pkl', 'wb') as f:
        pickle.dump(gData, f, pickle.HIGHEST_PROTOCOL)