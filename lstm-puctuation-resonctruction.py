#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fofe_punctuation_resonctruction import *

class LSTMTagger(nn.Module):

	def __init__(self, char2vec, hidden_dim, tagset_size, minibatch_size):
		super(LSTMTagger, self).__init__()
		self.hidden_dim = hidden_dim
		self.layer_size = 2 # 2 for number of layers, hardcode for now

		vocab_size = char2vec.shape[0]
		embedding_dim = char2vec.shape[1]
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.word_embeddings.weight = nn.Parameter(torch.from_numpy(char2vec))

		self.lstm = nn.LSTM(embedding_dim, hidden_dim, self.layer_size, batch_first=True, bidirectional=True)

		# The linear layer that maps from hidden state space to tag space
		self.hidden2tag = nn.Linear(hidden_dim * 2 * 2, tagset_size) # 2 for bidirectional, and 2 for concatenate
		self.hidden = self.init_hidden(minibatch_size)

	def init_hidden(self, minibatch_size):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (autograd.Variable(torch.zeros(self.layer_size * 2, minibatch_size, self.hidden_dim)),
				autograd.Variable(torch.zeros(self.layer_size * 2, minibatch_size, self.hidden_dim)))

	def forward(self, sentences):
		# sentences is 2D tensor
		n_sentence, context_size = sentences.size(0), sentences.size(1)
		embeds = self.word_embeddings(sentences)

		lstm_out, self.hidden = self.lstm(embeds, self.hidden)
		middle = context_size / 2 - 1

		# projection = lstm_out[:,middle:middle+2,:].view(n_sentence, -1)
		left_context = lstm_out[:,middle,:]
		right_context = lstm_out[:,middle+1,:]
		projection = torch.cat((left_context,right_context),1)

		tag_space = self.hidden2tag(projection)
		tag_scores = F.log_softmax(tag_space)
		return tag_scores

	def infer(self, leftContext, rightContext):
		sentence_in = autograd.Variable(
			torch.from_numpy(numpy.concatenate((leftContext, rightContext),axis=1)).type(torch.LongTensor)
		)
		tag_scores = model(sentence_in)
		values, indices = torch.max(tag_scores, 1)
		return indices


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
	parser.add_argument( '--context_size', type = int, default = 32,
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

	HIDDEN_DIM = 256 # TODO: tune this
	NUM_CLASS = 4 # four types of punctuation

	model = LSTMTagger(char2vec, HIDDEN_DIM, NUM_CLASS, args.batch_size)
	loss_function = nn.NLLLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.1)

	for epoch, f in enumerate(ifilter(lambda f: int(f[-2:]) < 12, filelist)):
		filename = os.path.join( args.src_dir, f )
		train = BatchConstructor( filename, punc2idx, char2idx )
		logger.info( '%s loaded' % f )

		for leftContext, rightContext, separator in \
				train.miniBatch( batchSize = args.batch_size, 
								 contextSize = args.context_size,
								 shuffle = True ): 
			if leftContext.shape[0] == args.batch_size:
				model.zero_grad()
				model.hidden = model.init_hidden(args.batch_size)

				sentence_in = autograd.Variable(
					torch.from_numpy(numpy.concatenate((leftContext, rightContext),axis=1)).type(torch.LongTensor)
				)

				# tag_scores = model(torch.cat(leftContext, rightContext))
				tag_scores = model(sentence_in)

				loss = loss_function(tag_scores, autograd.Variable(torch.from_numpy(separator)))
				loss.backward()
				optimizer.step()
		break

		 if (epoch + 1) % 11 == 0:
            confMat = numpy.zeros([len(punc2idx), len(punc2idx)])
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