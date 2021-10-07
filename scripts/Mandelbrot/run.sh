#!/bin/bash

julia -t4 mandel.jl | tee output.txt

ristretto *.png


