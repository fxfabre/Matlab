%%  Régression logistique simple (2 classes) et multi-classes
clear all;
close all;

%%  Initialisation data
% x < 2 => groupe 1
% x > 0 => groupe 2
% Attention, les id des groupes doivent être strictement positifs
data = load('data/data1D.txt');
x_train = data(:,1);
y_train = data(:,2);

%%  Train model
[B,dev,stat] = mnrfit(x_train, y_train);

%%  Eval "a la main"
N = size(x_train, 1);
x_test = [ones(N,1) x_train];

y_test = x_test * B; % nombres reels
y_test = -sign(y_test); % -1 ou 1
y_test = 0.5 * (y_test +3); % 1 ou 2
[y_train y_test]

%%  Résultats :
% 1 erreur
% influence du point extrême ?

%%  Classification avec mnrval(...)
% Classification de l'ensemble d'apprentissage
x_test = x_train;
y = mnrval(B, x_train);


%%  Idem, avec suppression de l'outlier
x_train = data(1:end-1 , 1);
y_train = data(1:end-1 , 2);

B = mnrfit(x_train, y_train);

N = size(x_train, 1);
x_test = [ones(N,1) x_train];

y_test = x_test * B; % nombres reels
y_test = -sign(y_test); % -1 ou 1
y_test = 0.5 * (y_test +3); % 1 ou 2
[y_train y_test]

%%
% Toujours 1 erreur, mais sur un point différent.
