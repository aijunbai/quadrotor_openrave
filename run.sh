#!/bin/bash

source "conf/config.sh"

PROFILE="False"
VERBOSE="False"
TEST="False"

while getopts "pvt" opt; do
    case "$opt" in
        p) PROFILE="True";;
        v) VERBOSE="True";;
        t) TEST="True";;
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

if [ $TEST = "True" ]; then
    QUADROTOR="$QUADROTOR --test"
fi

if [ $PROFILE = "True" ]; then
    $PYTHON -m cProfile -o $QUADROTOR_PROF $QUADROTOR
    profile $QUADROTOR_PROF
else
    $PYTHON $QUADROTOR
fi
