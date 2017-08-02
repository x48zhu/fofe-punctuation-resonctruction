#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python
# -*- coding: utf-8 -*-

import codecs
import argparse
import os
import re
import sys
import logging
import numpy

import tensorflow as tf

from itertools import imap, ifilter
from tqdm import tqdm
from hanziconv import HanziConv

logger = logging.getLogger()


################################################################################

# punc2idx = {
#     u'<space>' : 0,
#     u'<non-space>' : 1,
#     u'<other>' : 2,
#     # add the fine-grained classes you want, e.g.
#     # u'。' : 3,
#     # u'，' : 4,
#     # u'？' : 5
# }

# 0 is reserved
punc2idx = {
    u'。' : 1,
    '.': 1,
    u'，' : 2,
    ',': 2,
    u'？' : 3,
    '?': 3,
    u'！': 4,
    '!': 4
}



################################################################################


class BatchConstructor( object ):
    def __init__( self, basename, punc2idx, char2idx ):
        with codecs.open( '%s.char' % basename, 'rb', 'utf8' ) as char_file:
            char = char_file.read().strip().split()

        with codecs.open( '%s.sep' % basename, 'rb', 'utf8' ) as sep_file:
            sep = sep_file.read().strip().split(u'\n')

        assert len(sep) - len(char) == 1, '%s: incorrect preprocessing' % basename

        number =  r'^(?=[^A-Za-z]+$).*[0-9].*$'.encode('utf8')

        c_unk = char2idx[u'<unk>']
        self.char = [ char2idx.get(c, c_unk) for c in \
                      imap( lambda c: u'<numeric>' if re.match(number, c) else c, char ) ]

        # s_other = punc2idx['<other>']
        # self.sep = numpy.asarray([ punc2idx.get(s, s_other) for s in sep ], dtype = numpy.int32)

        s_other = 0
        target = ''.join(punc2idx.keys())
        # self.sep = numpy.asarray([ punc2idx[t] if t in s else s_other for t in target for s in sep ], dtype = numpy.int32)

        temp = []
        for s in sep:
            target_found = False
            for t in target:
                if t in s:
                    temp.append(punc2idx[t])
                    target_found = True
                    break
            if not target_found:
                temp.append(s_other)
        self.sep = numpy.asarray(temp)


    def __len__( self ):
        return len(self.sep)


    def miniBatch( self, batchSize = 256, contextSize = 32, shuffle = False ):
        padding = numpy.zeros( contextSize, dtype = numpy.int32 )
        chars = numpy.concatenate(
            [ padding,
              numpy.asarray( self.char, dtype = numpy.int32 ),
              padding ],
            0
        )

        leftBuff = numpy.ndarray( (batchSize, contextSize), dtype = numpy.int32 )
        rightBuff = numpy.ndarray( (batchSize, contextSize), dtype = numpy.int32 )

        order = numpy.arange( len(self.sep) )
        if shuffle:
            numpy.random.shuffle( order )

        nPassed = 0
        while nPassed < len(self.sep):
            nextBatch = order[nPassed: nPassed + batchSize]
            nNextBatch = nextBatch.shape[0]

            separator = self.sep[ nextBatch ]

            for i, j in enumerate(nextBatch):
                leftBuff[i] = chars[j: j + contextSize]
                rightBuff[i] = chars[j + contextSize: j + 2 * contextSize]

            yield leftBuff[:nNextBatch], rightBuff[:nNextBatch], separator

            nPassed += batchSize


################################################################################


def LoadChar2Vec( basename ):
    with codecs.open( '%s.wordlist' % basename, 'rb', 'utf8' ) as wordlist:
        idx2char = [u'<padding>'] + wordlist.read().strip().split()
        char2idx = dict( (c, i) for (i, c) in enumerate(idx2char) )

    with open( '%s.word2vec' % basename, 'rb' ) as word2vec:
        shape = numpy.fromfile( word2vec, dtype = numpy.int32, count = 2 )
        char2vec = numpy.concatenate(
            [ numpy.zeros((1, shape[1]), dtype = numpy.float32 ),
              numpy.fromfile( word2vec, dtype = numpy.float32 ).reshape( shape ) ],
            0
        )

    return char2vec, char2idx


################################################################################


class CmnSegmenter( object ):
    def __init__( self, char2vec, alpha = 0.7 ):
        self.graph = tf.Graph()

        gpu_option = tf.GPUOptions( per_process_gpu_memory_fraction = 0.96 )
        self.session = tf.Session( 
            config = tf.ConfigProto( gpu_options = gpu_option ),
            graph = self.graph
        )

        self.non_pretrained_params = []

        with self.graph.as_default():
            # debug
            # a = tf.constant( numpy.asarray([[[1,2,3]],[[1,2,3]]]), dtype = tf.float32 )
            # b = tf.constant( numpy.arange(24).reshape(2, 3, 4), dtype = tf.float32 )
            # print self.session.run( tf.matmul(a, b) )

            self.__initInput( alpha )
            self.__initConnection( char2vec )

            self.adam_op = tf.train.AdamOptimizer(  
                learning_rate = self.lr,
                epsilon = 1e-8
            ).minimize( self.cost, var_list = self.non_pretrained_params )

            self.sgd_op = tf.train.GradientDescentOptimizer(
                learning_rate = self.lr
            ).minimize( self.cost, var_list = self.non_pretrained_params )

            self.char_vec_op = tf.train.GradientDescentOptimizer(
                learning_rate = self.lr / 2
            ).minimize( self.cost, var_list = [ self.char2vec ] )

            init_op = tf.global_variables_initializer()

        self.session.run( init_op )


    def __initInput( self, alpha ):
        # batchSize x contextSize
        self.leftContext = tf.placeholder( tf.int32, [None, None] )

        # batchSize x contextSize
        self.rightContext = tf.placeholder( tf.int32, [None, None] )

        # batchSize
        self.target = tf.placeholder( tf.int32, [None] )

        self.lr = tf.placeholder( tf.float32, [] )
        self.keepProb = tf.placeholder( tf.float32, [] )

        self.alpha = tf.constant( 
            numpy.tile(
                numpy.float32(alpha) ** numpy.arange(1024), 
                [1024, 1]
            ),
            dtype = tf.float32 
        )


    def __initConnection( self, char2vec ):
        layerSize = [ char2vec.shape[1] * 2 ] + [ 512 ] * 3 + [ len(punc2idx) ]

        self.char2vec = tf.Variable( char2vec )

        batchSize, contextSize = tf.unstack( tf.shape( self.leftContext ) )

        rightAlpha = tf.reshape( self.alpha[:batchSize, :contextSize], [batchSize, 1, -1] )
        leftAlpha = tf.reshape( rightAlpha[:,::-1], [batchSize, 1, -1] )

        leftContext = tf.squeeze( 
            tf.matmul( leftAlpha, tf.gather( self.char2vec, self.leftContext ) ), 
            squeeze_dims = [1] 
        )
        rightContext = tf.squeeze( 
            tf.matmul( rightAlpha, tf.gather( self.char2vec, self.rightContext ) ), 
            squeeze_dims = [1] 
        )

        projection = tf.concat( [ leftContext, rightContext ], 1 )
        currentOut = tf.nn.dropout( projection, self.keepProb )

        for i in xrange( len(layerSize) - 1 ):
            inSize, outSize = layerSize[i], layerSize[i + 1]

            rng = numpy.float32(2.5 / numpy.sqrt(inSize + outSize))
            W = tf.Variable( 
                tf.random_uniform(
                    [inSize, outSize],
                    minval = -rng,
                    maxval = rng
                ) 
            )
            b = tf.Variable( tf.zeros( [outSize] ) )
            self.non_pretrained_params.extend( [W, b] )

            currentOut = tf.add( tf.matmul( currentOut, W ), b )
            if i != len(layerSize) - 2:
                currentOut = tf.nn.dropout( tf.nn.relu( currentOut ), self.keepProb )

        self.cost = tf.reduce_mean( 
            tf.nn.sparse_softmax_cross_entropy_with_logits( 
                logits = currentOut, 
                labels = self.target 
            )
        )

        self.inference = tf.argmax( currentOut, axis = -1 )
        self.test = tf.reduce_sum( 
            tf.to_int64( 
                tf.equal( 
                    self.inference, 
                    tf.to_int64( self.target )
                ) 
            ) 
        )


    def trainAdam( self, leftContext, rightContext, separator,
                   lr = 0.0128, keepProb = 1 - 0.256 ):
        trainCost = self.session.run( 
            [ self.adam_op, self.char_vec_op, self.cost ],
            feed_dict = {
                self.leftContext : leftContext,
                self.rightContext : rightContext,
                self.target : separator,
                self.lr : lr,
                self.keepProb : keepProb
            }
        )
        return trainCost[-1]


    def trainSGD( self, leftContext, rightContext, separator,
                  lr = 0.0128, keepProb = 1 - 0.256 ):
        trainCost = self.session.run( 
            [ self.sgd_op, self.char_vec_op, self.cost ],
            feed_dict = {
                self.leftContext : leftContext,
                self.rightContext : rightContext,
                self.target : separator,
                self.lr : lr,
                self.keepProb : keepProb
            }
        )
        return trainCost[-1]



    def infer( self, leftContext, rightContext ):
        predicted = self.session.run(
            self.inference,
            feed_dict = {
                self.leftContext : leftContext,
                self.rightContext : rightContext,
                self.keepProb : 1
            }
        )
        return predicted


    def accuracy( self, leftContext, rightContext, separator ):
        nCorrect = self.session.run(
            self.test,
            feed_dict = {
                self.leftContext : leftContext,
                self.rightContext : rightContext,
                self.target : separator,
                self.keepProb : 1
            }
        )
        return nCorrect

################################################################################


def EvalTest( segmenter, data, batchSize=128 ):
    nCorrect = 0
    for leftContext, rightContext, separator in \
        data.miniBatch( batchSize = batchSize, shuffle = False ):
        nCorrect += segmenter.accuracy( 
            leftContext, 
            rightContext, 
            separator 
        )
    return nCorrect


################################################################################

def ProcessConfusion (confMat,segmenter, data, batchSize=128):
    
    for leftContext, rightContext, separator in data.miniBatch(batchSize = batchSize, shuffle = False):
        predict = segmenter.infer(leftContext, rightContext)
        for i, j in zip(separator, predict):
            confMat[i, j] = confMat[i, j] + 1
    return confMat

def calculatePrecisionRecall (confMat):
    totalPrecision = numpy.sum(confMat, axis = 0)
    totalRecall = numpy.sum(confMat, axis = 1)
    trueVal = confMat.diagonal()
    avgPrecision = numpy.mean(numpy.divide(trueVal, totalPrecision))
    avgRecall = numpy.mean(numpy.divide(trueVal, totalRecall))
    F1 = 2.0*(avgPrecision*avgRecall)/(avgPrecision+avgRecall)
    return avgPrecision, avgRecall, F1



if __name__ == '__main__':
    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                         level = logging.INFO )

    parser = argparse.ArgumentParser()
    parser.add_argument( 'src_dir', type = str, 
                         help = 'directory containing the preprocessed files from conver-file.py' )
    parser.add_argument( 'char2vec_base', type = str,
                         help = 'e.g. /eecs/research/asr/mingbin/cleaner/word2vec/wiki-cmn-char' )
    parser.add_argument( '--alpha', type = float, default = 0.7,
                         help = 'forgetting factor' )
    parser.add_argument( '--context_size', type = int, default = 64,
                         help = 'how many characters to look ahead' )
    parser.add_argument( '--batch_size', type = int, default = 256 )
    parser.add_argument( '--learning_rate', type = float, default = 0.0128 )
    parser.add_argument( '--algorithm', type = str, default = 'sgd', 
                         choices = ['adam', 'sgd'] )

    args = parser.parse_args()
    logger.info( args )

    char2vec, char2idx = LoadChar2Vec( args.char2vec_base )
    logger.info( 'number of characters/words: %d' % len(char2idx) )
    logger.info( 'embedding size: [%d, %d]' % char2vec.shape )

    filelist = sorted(
        [ f[:f.rfind('.char')] for f in os.listdir( args.src_dir ) if f.endswith( '.char' ) ]
    )
    logger.info( 'number of training files: %d' % len(filelist) )
    logger.info( 'e.g. %s' % str(filelist[:10]) )


    segmenter = CmnSegmenter( char2vec, args.alpha )
    lr = args.learning_rate

    for epoch, f in enumerate(ifilter(lambda f: int(f[-2:]) < 12, filelist)):
        ####################
        # Train file by file

        filename = os.path.join( args.src_dir, f )
        train = BatchConstructor( filename, punc2idx, char2idx )
        logger.info( '%s loaded' % f )

        pbar = tqdm( total = len(train) )
        cnt, cost = 0, 0

        if args.algorithm == 'sgd':
            trainer = segmenter.trainSGD 
        elif args.algorithm == 'adam':
            trainer = segmenter.trainAdam
        else:
            raise NotImplementedError( 'hopelessness is your end' ) 

        for leftContext, rightContext, separator in \
                train.miniBatch( batchSize = args.batch_size, 
                                 contextSize = args.context_size,
                                 shuffle = True ): 
            if leftContext.shape[0] == args.batch_size:
                cost += trainer( 
                    leftContext, 
                    rightContext, 
                    separator,
                    lr = lr
                ) * leftContext.shape[0]

                cnt += leftContext.shape[0]
                pbar.update( leftContext.shape[0] )

        pbar.close()
        logger.info( '%s trained, avg-cost == %f' % (f, cost / cnt) )
        lr *= 0.5 ** (1./128)

        ##############
        # See accuracy 
        if (epoch + 1) % 11 == 0:
            dim = len(set(punc2idx.values()))
            confMat = numpy.zeros([dim, dim])
            testlist = list( ifilter(lambda f: int(f[-2:]) == 12, filelist) )
            ff = testlist[epoch % len(testlist)]
            filename = os.path.join( args.src_dir, ff )

            test = BatchConstructor( filename, punc2idx, char2idx )
            logger.info( '%s loaded' % ff )
            nCorrectTrain = EvalTest (segmenter, train)
            nCorrectTest = EvalTest( segmenter, test )
            confMat = ProcessConfusion(confMat, segmenter, test)
            precison, recall, f1 = calculatePrecisionRecall(confMat)
            logger.info('the confusion matrix is ')
            logger.info(confMat)
            logger.info('accuracy of Train %s: %d / %d' % (f, nCorrectTrain, len(train) ))
            logger.info( 'accuracy of Test %s: %d / %d' % (ff, nCorrectTest, len(test)) )
            logger.info( 'precision is '+str(precison))
            logger.info( 'recall is '+str(recall))
            logger.info( 'F1 is '+str(f1))
            logger.info( 'test Accuracy '+str(float(nCorrectTest)/len(test)))
    logger.info('Training and Testing Complete')




