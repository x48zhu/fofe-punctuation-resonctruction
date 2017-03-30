#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import string
import zhon.hanzi
import re
import codecs
import argparse
import logging
from hanziconv import HanziConv


logger = logging.getLogger()


if __name__ == '__main__':
	logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                         level = logging.INFO )

	parser = argparse.ArgumentParser()
	parser.add_argument( 'rspecifier', type = str, help = 'path to input file' )
	parser.add_argument( 'wspecifier', type = str, help = 'basename of output files' )

	args = parser.parse_args()
	logger.info( args )

	cmn_punc = zhon.hanzi.punctuation
	eng_punc = unicode(string.punctuation, 'utf8')
	sep = u'[%s]+' % (cmn_punc + eng_punc + u' ' + u'\n' )
	new_line_only = u'[%s]+' % u'\n'

	logger.info( sep )

	with codecs.open( args.rspecifier, 'rb', 'utf-8' ) as rspecifier:
		data = rspecifier.read().strip()

	pos = []
	for m in re.finditer( sep, data ):
		pos.append( m.start() )
		pos.append( m.end() )

	if pos[0] != 0:
		pos = [0] + pos
		isSep = False
	else:
		isSep = True

	if pos[-1] != len(data):
		pos.append( len(data) )
		endWithSep = False
	else:
		endWithSep = True


	write2char, write2sep = [], [] if isSep else [ '<non-space>' ]


	for i in xrange( len(pos) - 1 ):
		start, end = pos[i], pos[i + 1]
		token = data[start:end]
		if isSep:
			if re.match( new_line_only, token ):
				write2sep.append( u'<new-line>' )
			elif token == u' ':
				write2sep.append( u'<space>' )
			else:
				write2sep.append( re.sub( new_line_only, u'', token.strip() ) )
		else:
			has_chinese = any( u'\u4e00' <= c <= u'\u9fff' for c in token )
			if not has_chinese:
				write2char.append( token )
			else:
				for c in token:
					write2char.append( c )
				write2sep.extend( [u'<non-space>'] * (len(token) - 1) )
		isSep = not isSep


	with codecs.open( '%s.char' % args.wspecifier, 'wb', 'utf-8' ) as char_file:
		for i in xrange( len(write2char) ):
			write2char[i] = HanziConv.toSimplified( write2char[i] )
		char_file.write( u'\n'.join( write2char ) )

	with codecs.open( '%s.sep' % args.wspecifier, 'wb', 'utf-8' ) as sep_file:
		if not endWithSep:
			write2sep.append( 'non-space' )
		sep_file.write( u'\n'.join( write2sep ) )


