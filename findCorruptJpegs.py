#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
import os
import sys

from more_itertools import chunked
from tqdm import tqdm
import tensorflow as tf


def check_image( args ):
    filename, contents = args

    try:
        tf.io.decode_jpeg( contents )
    except Exception as e:
        return filename, e

if __name__ == '__main__':
    folder = sys.argv[1]
    outfile = sys.argv[2]

    assert os.path.isdir( folder )
    assert not os.path.exists( outfile )

    # Check files using tf.io.decode_jpeg
    with tqdm( unit = " files" ) as progress_bar, multiprocessing.Pool() as pool, open( outfile, 'wt' ) as output:
        for root, dirnames, filenames in os.walk( folder ):
            # File contents are read sequentially (to not spam a slow HDD with hundres of IOPS)
            # because the arguments in the generator expression are evaluated by the pool process serially
            # Then, the loaded data in RAM is processed in parallel.
            results = pool.imap( check_image,
                                 ( ( filename, open( os.path.join( root, filename ), 'rb' ).read() )
                                   for filename in filenames ) )

            for result in results:
                if result:
                    filename, exception = result
                    print()
                    print( filename, str( exception ) )
                    print()
                    output.write( filename + ' ' + str( exception ) + '\n' )
                progress_bar.update( 1 )
