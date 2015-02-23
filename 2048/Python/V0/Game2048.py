#!/usr/bin/python3

__all__= ("Game2048",)

from sys import stderr

NB_ROW = NB_COL= 4



class Game2048( ):
    __slots__= ("_array",)
    
    def __init__( self, array:tuple=() ) -> None:
        if not (array):
            array= [0 for i in range(NB_ROW * NB_COL)]
        
        assert (len(array) == NB_ROW * NB_COL), "Taille incorrecte"
        self._array = array
        return
    
    def __repr__( self ):
        return "<object {:s}({:s})>".format(self, self._array)
    
    def __str__( self ):
        array= self._array
        text= ""
        for i in range(0, NB_ROW*NB_COL, NB_COL):
            text+= " " + " | ".join(array[i:i+NB_COL])
        return text
    
    def Play( self ) -> None:

        return


def main( n=4 ) -> int:
    n = int(n)
    try:
        g= Game2048( [0 for i in range(n*n)] )
        g.Play()
    except Exception as ex:
        stderr.write( "Unexpected exception : " + str(ex) )
        return 1

    return 0


if (__name__ == "__main__"):
    exit(main(4))
