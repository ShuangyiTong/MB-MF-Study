/* 
  Shuangyi Tong <s9tong@edu.uwaterloo.ca>
  Oct 11, 2018

  C++ implementation of forward
*/
#include <stddef.h>
#include <signal.h>
#include <cstring>
#include <vector>
#include <cmath>
#include <boost/python.hpp>
/*
  The performance of this array wrapper (tested locally) is almost the same as numpy array
  performance (i.e. around 4x slower than python native list, tested by stack overflow user)
  https://stackoverflow.com/questions/29281680/numpy-individual-element-access-slower-than-for-lists
*/
#include "array_wrapper.h"

class Forward {
  double*** _Transition;
  double**  _Q_values;
  int* _state_iterate_buf;
  int _num_states, _num_actions, _num_trans_prob, _output_offset = 0;
  double _epsilon, _learning_rate, _discount_factor, _num_children = 0.f;

  // Fill state_iterate_buf with states indices to iterate
  inline void getMDPChild(int state, int action) {
    int first_child = state * static_cast<int>(_num_children) + action * _num_trans_prob + 1;
    for (int i = 0; i < _num_trans_prob; ++i) {
      _state_iterate_buf[i] = first_child + i ;
    }
  }

  // Generate one Q value given the state and action
  inline void generateSingleQ(int state, int action) {
    double Q_value = 0;
    getMDPChild(state, action); // Refresh state iterate buffer
    for (int next_state_ind = 0; next_state_ind < _num_trans_prob; ++next_state_ind) {
      int next_state = _state_iterate_buf[next_state_ind];
      double Q_value_next_state;
      if (next_state < _output_offset) { // internal leaf
        double argmax_value = _Q_values[next_state][0];
        for (int next_state_next_action = 1; next_state_next_action < _num_actions; 
             ++next_state_next_action) {
          argmax_value = _Q_values[next_state][next_state_next_action] > argmax_value?
                         _Q_values[next_state][next_state_next_action] : argmax_value;
        }
        Q_value_next_state = argmax_value;
      }
      else {
        Q_value_next_state = _reward_array[next_state - _output_offset];
      }
      Q_value += _Transition[state][action][next_state_ind] * Q_value_next_state;
    }
    _Q_values[state][action] = Q_value;
  }

  inline int getMDPParent(int state) {
    return ceil(state / _num_children) - 1;
  }

  inline int getAction(int state) {
    return floor(((state - 1) % (long long)_num_children) / _num_actions);
  }

  inline int getStateIndex(int state) {
    return (state + 1) % 2;
  }

  void updateQ(int state, int action) {
    generateSingleQ(state, action);
    if (state == 0) {
      return;
    }
    else {
      updateQ(getMDPParent(state), getAction(state));
    }
  }

public:
  // buffer used to exchange data between python interpreter and this class
  int _reward_array[BUFFER_SIZE];
  double _Q_value_buf[BUFFER_SIZE];

  Forward(int num_states, int num_actions, int num_trans_prob, int output_offset,
          double epsilon, double learning_rate, double discount_factor)
    : _num_states (num_states)
    , _num_actions (num_actions)
    , _num_trans_prob (num_trans_prob)
    , _output_offset (output_offset)
    , _epsilon (epsilon)
    , _learning_rate (learning_rate)
    , _discount_factor (discount_factor) {
    _state_iterate_buf = new int[_num_trans_prob];
    // Build transition matrix
    _Transition = new double**[_num_states];
    for (int state = 0; state < _num_states; ++state) {
      _Transition[state] = new double*[_num_actions];
      for (int action = 0; action < _num_actions; ++action) {
        _Transition[state][action] = new double[_num_trans_prob];
        for (int next_state = 0; next_state < _num_trans_prob; ++next_state) {
          _Transition[state][action][next_state] = 1.f / _num_trans_prob;
        }
      }
    }
    _Q_values = new double*[_output_offset];
    for (int state = 0; state < _output_offset; ++state) {
      _Q_values[state]   = new double[_num_actions]();
    }
    _num_children = _num_actions * _num_trans_prob;
  }
  
  ~Forward() {
    for (int i = 0; i < _output_offset; ++i) {
      delete _Q_values[i];
    }
    delete _Q_values;
    for (int i = 0; i < _num_states; ++i) {
      for(int j = 0; j < _num_actions; ++j) {
        delete _Transition[i][j];
      }
      delete _Transition[i];
    }
    delete _Transition;
    delete _state_iterate_buf;
  }

  void generateQ() {
    // Generate Q value from high state index to low ones
    for (int state = _output_offset - 1; state >= 0; --state) {
      for (int action = 0; action < _num_actions; ++action) {
        generateSingleQ(state, action);
      }
    }
  }

  double optimize(int state, int action, int next_state) {
#ifdef DEBUG
    assert( state == getMDPParent(next_state) );
    assert( action == getAction(next_state) );
#endif
    int next_state_ind = getStateIndex(next_state);
    double spe = 1 - _Transition[state][action][next_state_ind];
    for (int post_state_ind = 0; post_state_ind < _num_trans_prob; ++post_state_ind) {
      if (post_state_ind == next_state_ind) {
        _Transition[state][action][post_state_ind] += _learning_rate * spe; 
      }
      else { 
        _Transition[state][action][post_state_ind] *= 1 - _learning_rate;
      }
    }
    updateQ(state, action);
    return spe;
  }

  void fillQValueBuffer(int state) {
    memcpy(_Q_value_buf, _Q_values[state], _num_actions);
  }
};

BOOST_PYTHON_MODULE(cforward) {
  using namespace boost::python;

  class_<Forward>("cForward", init<int, int, int, int, double, double, double>())
    .add_property("reward_array", wrap_array(&Forward::_reward_array))
    .add_property("Q_buf", wrap_array(&Forward::_Q_value_buf))
    .def("generate_Q", &Forward::generateQ)
    .def("optimize", &Forward::optimize)
    .def("fill_Q_value_buf", &Forward::fillQValueBuffer)
  ;
}