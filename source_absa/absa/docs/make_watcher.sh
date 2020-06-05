#!/bin/bash

check() {
    dir="$1"
    chsum1=`checksumdir -a md5 $dir`
    sleep 3
    chsum2=`checksumdir -a md5 $dir`
    if [ "$chsum1" != "$chsum2" ]
    then
      echo "Source folder changed"
      eval "make html"
      # inject javascript into all html files
      sed -i '8i <script type="text/javascript" src="http://livejs.com/live.js"></script>' ./build/html/index.html
    fi
}
while true
do
  check $*
done