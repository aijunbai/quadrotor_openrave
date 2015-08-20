#!/bin/bash

source "conf/config.sh"

PROFILE="False"
VERBOSE="False"

while getopts "pv" opt; do
    case "$opt" in
        p) PROFILE="True";;
        v) VERBOSE="True";;
        *) ;;
    esac
done


profile() {
    PNG=`mktemp`
    $PYTHON gprof2dot.py -f pstats $1 | dot -Tpng -o $PNG
    eog $PNG
    rm -f $QUADROTOR_PROF
    rm -f $PNG
}

if [ $VERBOSE = "True" ]; then
    QUADROTOR="$QUADROTOR --verbose"
fi

if [ $PROFILE = "True" ]; then
    time $PYTHON -m cProfile -o $QUADROTOR_PROF $QUADROTOR
    profile $QUADROTOR_PROF
else
    time $PYTHON $QUADROTOR
fi
