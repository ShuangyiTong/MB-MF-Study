import getopt
import sys
import csv
import os
import simulation as sim
import analysis
import dill as pickle # see https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions

from analysis import gData, MODE_MAP
from tqdm import tqdm
from numpy.random import choice

usage_str = """
Model-free, model-based learning simulation

Usage:

Simulation control parameters:

    -d load parameters from csv file, default is regdata.csv

    -n [number of parameters entries to simulate]

    --episodes [num episodes]

    --trials [num trials per episodes]

    --set-param-file [parameter file]             Specify the parameter csv file to load

    --mdp-stages [MDP environment stages]         Specify how many stages in MDP environment

    --ctrl-mode <min-spe/max-spe/min-rpe/max-rpe/min-mf-rel/max-mf-rel/min-mb-rel/max-mb-rel> 
                                                  Choose control agent mode

    --legacy-mode                                 Use legacy MDP environment, which treats one type of terminal reward as one
                                                  state. Since C++ ext is not implemented for this type of environment, legacy
                                                  pure Python implemenation for FORWARD will be use.

    --disable-control                             Disable control agents

    --all-mode                                    Execute all control mode

    --enable-static-control                       Use static control instead of DDQN control

    --disable-detail-plot                         Disable plot for each simulation

    --disable-c-ext                               Disable using C extension

    --less-control-input                          Less environment input for control agent

    --save-ctrl-rl                                Save control RL agent object for further use

Analysis control parameters:

    --re-analysis [analysis object pickle file]   Re-run analysis functions

    --PCA-plot                                    Generate plot against PCA results. Not set by default because
                                                  previous PCA run shows MB preference gives 99% variance, so comparing 
                                                  against MB preference is good enough, instead of some principal component

    --learning-curve-plot                         Plot learning curves

    --use-confidence-interval                     When plot with error bar, use confidence interval instead of IQR

    --separate-learning-curve                     Separate learning curve plot

    --disable-auto-max                            Use fix max value on y axis when plotting learning curve in one episode     

    --to-excel [subject id]                       Generate a excel file for specific subject with detail sequence of data

    --disable-action-compare                      Use actions as feature space

    --enable-score-compare                        Use score as feature space

    --human-data-compare                          Enable comparison against the full columns of human data

    --use-selected-subjects                       Use selected subjects, defualt is min 25 50 75 max five subjects

    --head-tail-subjects                          Use head and tail subjects to emphasize the difference

    --cross-mode-plot                             Plots that compare data between modes

    --enhance-compare <boost/inhibit/cor/sep>     Only plot two modes depending on the scenario to comapre

    --cross-compare [mode]                        Extract best action sequence from subject A in a given mode. Apply to subject B.
                                                  Plot against subject B's original data
    
    --sub-A [subject A]                           Subject A for cross compare

    --sub-B [subject B]                           Subject B for cross compare
"""

def usage():
    print(usage_str)

LOAD_PARAM_FILE   = False
NUM_PARAMETER_SET = 82
ALL_MODE          = False
ANALYSIS_OBJ      = None
TO_EXCEL          = None
SCENARIO          = None
CROSS_MODE_PLOT   = False
CROSS_COMPARE     = False
CROSS_COMPARE_MOD = 'min-spe'
SUBJECT_A         = 10 # Low MB->MF trans rate
SUBJECT_B         = 17 # High MB->MF trans rate
PARAMETER_FILE    = 'regdata.csv'

SCENARIO_MODE_MAP = {
    'boost'   : ['min-spe', 'min-rpe'],
    'inhibit' : ['min-spe', 'max-spe'],
    'cor'     : ['min-rpe-min-spe', 'max-rpe-max-spe'],
    'sep'     : ['min-rpe-max-spe', 'max-rpe-min-spe']
}

def reanalysis(analysis_object):
    with open(analysis_object, 'rb') as pkl_file:
        gData = pickle.load(pkl_file)
    if CROSS_MODE_PLOT:
        if SCENARIO is not None:
            gData.cross_mode_summary(SCENARIO_MODE_MAP[SCENARIO])
        else:
            gData.cross_mode_summary()
    elif CROSS_COMPARE:
        if CROSS_COMPARE_MOD == 'all':
            mode_iter_lst = [mode for mode, _ in MODE_MAP.items()]
        else:
            mode_iter_lst = [CROSS_COMPARE_MOD]
        for compare_mode in mode_iter_lst:
            for _ in range(100):
                SUBJECT_A, SUBJECT_B = choice(82, 2, replace=False)
                # set up simulation with static control sequence from subject A
                sim.STATIC_CONTROL_AGENT = True
                sim.TRIALS_PER_EPISODE = gData.trial_separation
                sim.TOTAL_EPISODES = 1
                sim.static_action_map[compare_mode] = gData.get_optimal_control_sequence(compare_mode, SUBJECT_A)
                sim.CONTROL_MODE = compare_mode
                sim.ENABLE_PLOT = False
                with open(PARAMETER_FILE) as f:
                    csv_parser = csv.reader(f)
                    param_list = []
                    for row in csv_parser:
                        param_list.append(tuple(map(float, row[:-1])))
                res_data_df, res_detail_df = sim.simulation(*(param_list[SUBJECT_B]), PARAMETER_SET=str(SUBJECT_B), return_res=True)
                gData.data[analysis.MODE_IDENTIFIER] = [None]
                gData.detail[analysis.MODE_IDENTIFIER] = [None]
                gData.data[analysis.MODE_IDENTIFIER][0] = res_data_df
                gData.detail[analysis.MODE_IDENTIFIER][0] = res_detail_df
                gData.cross_mode_summary([analysis.MODE_IDENTIFIER, compare_mode], [0, SUBJECT_B], [SUBJECT_A, SUBJECT_B])
                gData.plot_transfer_compare_learning_curve(compare_mode, SUBJECT_B, SUBJECT_A)
    else:
        for mode, _ in tqdm(MODE_MAP.items()):
            try:
                gData.set_current_mode(mode)
                gData.generate_summary(mode)
            except KeyError:
                print('mode: ' + mode + ' data not found. Skip')
    if TO_EXCEL is not None:
        gData.sequence_to_excel(TO_EXCEL)

if __name__ == '__main__':
    short_opt = "hdn:"
    long_opt  = ["help", "mdp-stages=", "disable-control", "ctrl-mode=", "set-param-file=", "trials=", "episodes=", "all-mode", "enable-static-control",
                 "disable-c-ext", "disable-detail-plot", "less-control-input", "re-analysis=", "PCA-plot", "learning-curve-plot", "use-confidence-interval",
                 "to-excel=", "disable-action-compare", "enable-score-compare", "use-selected-subjects", "save-ctrl-rl", "head-tail-subjects", 
                 "human-data-compare", "disable-auto-max", "legacy-mode", "separate-learning-curve", "cross-mode-plot", "cross-compare=", "sub-A=", "sub-B=",
                 "enhance-compare="]
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
        elif o == "--enable-static-control":
            sim.STATIC_CONTROL_AGENT = True
        elif o == "--disable-c-ext":
            sim.DISABLE_C_EXTENSION = True
        elif o == "--legacy-mode":
            sim.DISABLE_C_EXTENSION = True
            sim.LEGACY_MODE = True
        elif o == "--disable-detail-plot":
            sim.ENABLE_PLOT = False
        elif o == "--less-control-input":
            sim.MORE_CONTROL_INPUT = False
        elif o == "--save-ctrl-rl":
            sim.SAVE_CTRL_RL = True
        elif o == "--PCA-plot":
            analysis.PCA_plot = True
        elif o == "--learning-curve-plot":
            analysis.PLOT_LEARNING_CURVE = True
        elif o == "--separate-learning-curve":
            analysis.MERGE_LEARNING_CURVE = False
        elif o == "--use-confidence-interval":
            analysis.CONFIDENCE_INTERVAL = True
        elif o == "--disable-auto-max":
            analysis.LEARNING_CURVE_AUTO_MAX = False
        elif o == "--disable-action-compare":
            analysis.ACTION_COMPARE = False
        elif o == "--enable-score-compare":
            analysis.SOCRE_COMPARE = True
        elif o == "--human-data-compare":
            analysis.HUMAN_DATA_COMPARE = True
        elif o == "--use-selected-subjects":
            analysis.USE_SELECTED_SUBJECTS = True
        elif o == "--head-tail-subjects":
            analysis.HEAD_AND_TAIL_SUBJECTS = True
        elif o == "--to-excel":
            TO_EXCEL = int(a)
        elif o == "--re-analysis":
            ANALYSIS_OBJ = a
        elif o == "--cross-mode-plot":
            CROSS_MODE_PLOT = True
        elif o == "--enhance-compare":
            SCENARIO = a
        elif o == "--cross-compare":
            CROSS_COMPARE = True
            CROSS_COMPARE_MOD = a
        elif o == "--sub-A":
            SUBJECT_A = int(a)
        elif o == "--sub-B":
            SUBJECT_B = int(a)
        else:
            assert False, "unhandled option"

    if ANALYSIS_OBJ is not None:
        if os.path.isdir(ANALYSIS_OBJ):
            analysis_object_list = filter(lambda f: f.endswith('.pkl'), os.listdir(ANALYSIS_OBJ))
            parent_res_dir = analysis.RESULTS_FOLDER
            for index, obj in enumerate(analysis_object_list):
                analysis.RESULTS_FOLDER = parent_res_dir + '/subrun_' + str(index) + '/'
                reanalysis(os.path.join(ANALYSIS_OBJ, obj))
        else:
            reanalysis(ANALYSIS_OBJ)
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