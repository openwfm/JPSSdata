#!/bin/bash

if [ $(ls -l $1/*.hdf 2>/dev/null | wc -l) -gt 0 ]; then
	ln -s $1/*.hdf .
elif [ $(ls -l $1/satdata/*.hdf 2>/dev/null | wc -l) -gt 0 ]; then
	ln -s $1/satdata/*.hdf .
fi

if [ $(ls -l $1/*.h5 2>/dev/null | wc -l) -gt 0 ]; then
	ln -s $1/*.h5 .
elif [ $(ls -l $1/satdata/*.h5 2>/dev/null | wc -l) -gt 0 ]; then
	ln -s $1/satdata/*.h5 .
fi

if [ $(ls -l $1/*.nc 2>/dev/null | wc -l) -gt 0 ]; then
	ln -s $1/*.nc .
elif [ $(ls -l $1/satdata/*.nc 2>/dev/null | wc -l) -gt 0 ]; then
	ln -s $1/satdata/*.nc .
fi
