clear all;
close all;

%% Données presque planes, avec vecteurs propres confondus dans les axes
% Emplacement des points
%               X
%   X       X   X   X       X
%               X
A = [ 0 , 2 ,  0.1 ;
      2 , 2 , -0.1 ;
      3 , 0 , -0.1 ;
      3 , 2 ,  0.0 ;
      3 , 4 , -0.1 ;
      4 , 2 , -0.1 ;
      6 , 2 ,  0.1 ];
B = A - repmat(mean(A), 7, 1); % mean( B ) = 0
C = B ./ repmat(std(B), 7, 1); %  std( C ) = 1

% Matrice de correlation : C' * C / 6 =
%    1.0000         0         0
%         0    1.0000         0
%         0         0    1.0000

% 1° vecteur propre : 1, 0, 0
% 2° vecteur propre : 0, 1, 0
% 3° vecteur propre : 0, 0, 1

clear all;

%% Memes données, mais un peu penchées
% Emplacement des points
%               X
%   X       X   X   X       X
%               X
A = [ 0 , 2 ,  0.1 ;
      2 , 2 , -0.1 ;
      3 , 0 , -0.1 ;
      3 , 2 ,  0.0 ;
      3 , 4 ,  0.1 ;
      4 , 2 , -0.1 ;
      6 , 2 ,  0.1 ];
B = A - repmat(mean(A), 7, 1); % mean( B ) = 0
C = B ./ repmat(std(B), 7, 1); %  std( C ) = 1

% Matrice de correlation : C' * C / 6 =
%    1.0000         0         0
%         0    1.0000    0.5774
%         0    0.5774    1.0000

% 1° vecteur propre : 1, 0     , 0
% 2° vecteur propre : 0, 1     , 0.5774
% 3° vecteur propre : 0, 0.5774, 1

[corr_matrix, new_coord, eigenval] = princomp( A );

%% corr_matrix
% Matrice de correlation
% chaque colonne est un vecteur propre

%% new_coord
% données dans les nouvelles coordonnées
% Première  colonne : coordonnées suivant le 1° vecteur
% Deuxième  colonne : coordonnées suivant le 2° vecteur
% Troisième colonne : coordonnées suivant le 3° vecteur
%
% new_coord = B * corr_matrix
% Attention, on utilise bien B, pas C
% Il faut centrer les données, pas les réduire.

%% eigenval
% valeurs propres par ordre décroissant

plot( cumsum(eigenval) ./ sum(eigenval) );


