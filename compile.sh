clear;clear
gcc main.c -lm -Ofast -o fdgan
clang main.c -lm -Ofast -o cfdgan
gcc main.c -lm -o dgan
clang main.c -lm -o cdgan
