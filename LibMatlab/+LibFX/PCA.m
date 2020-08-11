classdef PCA
    %LIBFX_PCA Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (SetAccess = immutable)
        Data
        EigenValues
        EigenVectors
        NewCoordInput
    end
    
    methods
        function this = PCA(inputData)
            this.Data = inputData;
            
            [corr_matrix, new_coord, eigenval] = princomp( inputData );
            this.EigenValues = eigenval;
            this.EigenVectors = corr_matrix;
            this.NewCoordInput = new_coord;
        end
        
        function outData = ToNewCoordinates(this, inputData)
            N = size( inputData, 1);
            centerData = inputData - repmat(mean(inputData), N, 1);
            outData = centerData * this.EigenVectors;
        end
        
        function DisplayCumulativeEigenValue(this)
            total = sum(this.EigenValues);
            figure('Name', 'Somme cumulative valeurs propre');
            plot( cumsum(this.EigenValues) ./ total );
        end
        
        function nbComposantes = GetNbComposantesToKeep(this, totalVariance)
            sum_eigenval = cumsum(this.EigenValues) ./ sum(this.EigenValues);
            eigenval_significative = sum_eigenval(sum_eigenval < totalVariance);
            nbComposantes = size(eigenval_significative,1);
        end
        
        function data_X = KeepComposantes(this, nbComposantes)
            data_X = this.NewCoordInput(:, 1:nbComposantes);
        end

    end
end

% import matlab.unittest.TestSuite
% run(TestSuite.fromFolder('Projets/LibMatlab/UnitTests'))

