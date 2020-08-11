function [ data ] = CreateDataSet( std, N, probaLaw, fileName, nbOutliers )

    mean = 2;

    %% Génération données 1D linéairement séparable
    data_x = random(probaLaw, mean, std, [N, 1]);
    
    %% Ajout d'outliers
    outliers = random('Uniform', -100000, -10000, [nbOutliers, 1]);
    data_x = [data_x ; outliers];
    
    %% Classification des points
    idx_class1 = data_x <= mean;
    idx_class2 = data_x >  mean;

    data_y = ones(N, 1);
    data_y(idx_class2) = 2;

    %% Affichage données
    figure('Name', fileName);
    hold on;
    plot(data_x(idx_class1), 0, 'r*');
    plot(data_x(idx_class2), 0, 'b*');

    %% Sauvegarde données
    csvwrite(fileName, [data_x data_y]);
    
    %% Return data
    data = [ data_x data_y];
    
end

