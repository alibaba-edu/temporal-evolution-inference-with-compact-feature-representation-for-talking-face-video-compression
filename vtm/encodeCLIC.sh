#!/bin/sh

./vtm/bin/EncoderCLIC -c ./vtm/cfg/encoder_lowdelay_clic.cfg -c ./vtm/cfg/per-sequence/50.cfg -q $2 -i $1_org.yuv -o $1_rec.yuv -b $1.bin >>$1.log 
