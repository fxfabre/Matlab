classdef TestPCA < matlab.unittest.TestCase
    
    properties
    end
    
    methods (Test)
        function TestEigenVector(testCase)
            data = [0 , 2 ,  0.1 ;
                    2 , 2 , -0.1 ;
                    3 , 0 , -0.1 ;
                    3 , 2 ,  0.0 ;
                    3 , 4 , -0.1 ;
                    4 , 2 , -0.1 ;
                    6 , 2 ,  0.1 ];
            pca = LibFX.PCA(data);
            
            eigenVector = [1 , 0, 0 ; 0, 1, 0 ; 0, 0, 1];
            testCase.verifyEqual(pca.EigenVectors, eigenVector, 'AbsTol', sqrt(eps));
        end
        
        function TestCoordinates(testCase)
            coord = [2, 0 ; -2, 0 ; 0, 1 ; 0, -1];

            alpha = pi/6;
            rotateMatrix = [cos(alpha), -sin(alpha) ; sin(alpha), cos(alpha) ];

            newCoord = coord * rotateMatrix';

            pca = LibFX.PCA(newCoord);
          
            ExpectEigenVectors = [0.866025403784438, -0.5 ; 0.5, 0.866025403784438];
            testCase.verifyEqual(pca.EigenVectors, ExpectEigenVectors, 'AbsTol', sqrt(eps));
            testCase.verifyEqual(pca.NewCoordInput, coord, 'AbsTol', sqrt(eps));
        end
        
        function TestToNewCoordinates1(testCase)
            data = [0 , 2 ,  0.1 ;
                    2 , 2 , -0.1 ;
                    3 , 0 , -0.1 ;
                    3 , 2 ,  0.0 ;
                    3 , 4 , -0.1 ;
                    4 , 2 , -0.1 ;
                    6 , 2 ,  0.1 ];
            pca = LibFX.PCA(data);
            
            testCase.verifyEqual(pca.NewCoordInput, pca.ToNewCoordinates(data), 'AbsTol', sqrt(eps));
        end
        
        function TestToNewCoordinates2(testCase)
            coord = [2, 0 ; -2, 0 ; 0, 1 ; 0, -1];

            alpha = pi/6;
            rotateMatrix = [cos(alpha), -sin(alpha) ; sin(alpha), cos(alpha) ];

            newCoord = coord * rotateMatrix';

            pca = LibFX.PCA(newCoord);
          
            ExpectEigenVectors = [0.866025403784438, -0.5 ; 0.5, 0.866025403784438];
            
            testCase.verifyEqual(pca.EigenVectors, ExpectEigenVectors, 'AbsTol', sqrt(eps));
            testCase.verifyEqual(pca.NewCoordInput, coord, 'AbsTol', sqrt(eps));
            testCase.verifyEqual(pca.NewCoordInput, pca.ToNewCoordinates(newCoord), 'AbsTol', sqrt(eps));
        end
        
        function TestRandomData(testCase)
            N = 1000;
            X = random('normal', 0   , 1  , N, 1);
            Y = random('normal', 0   , 1  , N, 1);
            Z = random('unif'  , -100, 100, N, 1);
            
            coord = [X Y Z];
            
            pca = LibFX.PCA(coord);
            
            testCase.verifyEqual(pca.NewCoordInput, pca.ToNewCoordinates(coord), 'AbsTol', sqrt(eps));
        end
    end
    
end

