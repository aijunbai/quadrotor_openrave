#!/bin/bash - 
#===============================================================================
#
#          FILE: configure.sh
# 
#         USAGE: ./configure.sh 
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (), 
#  ORGANIZATION: 
#       CREATED: 08/18/2015 22:28
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

export TRAJOPT_LOG_THRESH="WARN"

PYTHON=`which python`
BASH=`which bash`

QUADROTOR="quadrotor.py"
QUADROTOR_PROF="quadrotor.prof"

