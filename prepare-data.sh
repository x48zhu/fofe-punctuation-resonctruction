#!/cs/local/bin/bash

export this_dir=$(cd $(dirname $0); pwd)
export src=${src:-/local/scratch1/mingbin/gigaword/cmn_gw_5/data/parsed-data}
export dst=${dst:-/local/scratch1/mingbin/punctuation-reconstruction}
export proc=${this_dir}/conver-file.py

for f in `ls ${src}`
do
	echo "${proc} ${src}/${f} ${dst}/${f}"
done | parallel 