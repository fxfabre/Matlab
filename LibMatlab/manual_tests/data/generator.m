clear all;
close all;

%% Constantes
mean = 2;
std  = 1;
NbPoint = 500;

%% Appel fonction
CreateDataSet(std, NbPoint, 'Normal', 'data1D.csv', 0);
CreateDataSet(std, NbPoint, 'Normal', 'data1D_noise.csv', 30);




