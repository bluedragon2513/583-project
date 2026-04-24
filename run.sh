#!/bin/bash
# Wrapper that sets the JVM library path before running project scripts.
# Usage:
#   ./run.sh index_wiki.py        -- build the Lucene index
#   ./run.sh search_jeopardy.py   -- run retrieval and evaluation

export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export LD_LIBRARY_PATH=$JAVA_HOME/lib/server:$JAVA_HOME/lib:$LD_LIBRARY_PATH

exec venv/bin/python3 "$@"
